import cv2
import time
import torch
import numpy as np
import Jetson.GPIO as GPIO
from ultralytics import YOLO

# =====================================================
# 기본 설정
# =====================================================
YOLO_MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
YOLO_IMGSZ = 320
MIDAS_INTERVAL = 1

# MiDaS 가까운 영역 기준
NEAR_THRESHOLD = 0.78
MIN_OBSTACLE_AREA = 1800
MAX_OBSTACLE_AREA_RATIO = 0.28
IGNORE_BOTTOM_RATIO = 0.82

MIN_BOX_W = 25
MIN_BOX_H = 35
MAX_BOX_WIDTH_RATIO = 0.75

# 연속 감지 조건
CENTER_CONFIRM_FRAMES = 2
SIDE_CONFIRM_FRAMES = 3

# =====================================================
# 객체별 위험 정책
# =====================================================
DYNAMIC_DANGER_OBJECTS = ["car", "bicycle", "kickboard"]
STATIC_OBSTACLES = ["bollard", "utility pole"]
NOTICE_OBJECTS = ["person", "tree"]

# unknown obstacle은 표시만 하고 진동하지 않음
ENABLE_UNKNOWN_VIBRATION = False

# 카메라 기반 진동은 보수적으로
VIB_COOLDOWN = 1.8

# 연속 감지 조건
CENTER_DYNAMIC_CONFIRM = 2      # 중앙의 차량/자전거/킥보드
CENTER_STATIC_CONFIRM = 2       # 중앙의 볼라드/전봇대
SIDE_DYNAMIC_CONFIRM = 999      # 좌우 차량은 사실상 진동 안 함
SIDE_STATIC_CONFIRM = 5         # 좌우 고정 장애물은 매우 보수적으로

last_vib_time = 0

near_confirm_count = 0
last_near_direction = None
last_near_label = None

# =====================================================
# GPIO 설정
# =====================================================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

# =====================================================
# 모델 로드
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

print("YOLO 로딩 중...")
yolo = YOLO(YOLO_MODEL_PATH)
print("YOLO 클래스:", yolo.names)

print("MiDaS 로딩 중...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.small_transform
print("MiDaS 로드 완료")


# =====================================================
# 진동 함수
# =====================================================
def all_vibrations_off():
    GPIO.output([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.LOW)


def direction_to_pin(direction):
    if direction == "LEFT":
        return VIB_LEFT
    if direction == "CENTER":
        return VIB_CENTER
    if direction == "RIGHT":
        return VIB_RIGHT
    return None


def trigger_vibration(direction, pattern="SHORT"):
    global last_vib_time

    now = time.time()
    if now - last_vib_time < VIB_COOLDOWN:
        return

    pin = direction_to_pin(direction)
    if pin is None:
        return

    all_vibrations_off()

    if pattern == "DOUBLE":
        for _ in range(2):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.12)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.08)
    else:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.15)
        GPIO.output(pin, GPIO.LOW)

    last_vib_time = time.time()


# =====================================================
# 방향 / 위험도
# =====================================================
def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "LEFT"
    elif x_center < frame_width * 2 / 3:
        return "CENTER"
    else:
        return "RIGHT"


def get_object_risk(label):
    if label in ["car", "bicycle", "kickboard"]:
        return "HIGH"
    if label in ["person", "bollard", "tree", "utility pole"]:
        return "MEDIUM"
    return "LOW"


# =====================================================
# MiDaS
# =====================================================
def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth)

    return (depth - depth_min) / (depth_max - depth_min)


def estimate_depth(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return normalize_depth(depth)


def find_near_obstacle_from_depth(depth_map, frame_width, frame_height):
    """
    MiDaS depth map에서 가까운 영역 후보를 찾는다.
    단, 넓은 바닥/벽/책상면 같은 영역은 최대한 제외한다.
    """

    valid_depth = depth_map.copy()

    # 화면 아래쪽은 바닥/자기 몸/노이즈로 간주해서 제외
    ignore_y = int(frame_height * IGNORE_BOTTOM_RATIO)
    valid_depth[ignore_y:, :] = 0

    near_mask = (valid_depth >= NEAR_THRESHOLD).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN, kernel)
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        near_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    frame_area = frame_width * frame_height

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_OBSTACLE_AREA:
            continue

        if area > frame_area * MAX_OBSTACLE_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue

        if w > frame_width * MAX_BOX_WIDTH_RATIO:
            continue

        aspect_ratio = w / max(h, 1)

        # 너무 가로로 긴 영역은 바닥/책상/벽 일부일 가능성이 큼
        if aspect_ratio > 4.0:
            continue

        x_center = x + w / 2
        direction = get_direction(x_center, frame_width)

        region = valid_depth[y:y + h, x:x + w]
        depth_score = float(np.mean(region))

        near_pixels = np.sum(region >= NEAR_THRESHOLD)
        total_pixels = region.size
        near_ratio = near_pixels / max(total_pixels, 1)

        if near_ratio < 0.25:
            continue

        candidates.append({
            "detected": True,
            "direction": direction,
            "box": (x, y, x + w, y + h),
            "area": area,
            "depth_score": depth_score,
            "near_ratio": near_ratio
        })

    if not candidates:
        return {
            "detected": False,
            "direction": None,
            "box": None,
            "area": 0,
            "depth_score": 0,
            "near_ratio": 0
        }

    def score(c):
        depth_part = c["depth_score"] * 0.55
        near_part = c["near_ratio"] * 0.30
        area_part = min(c["area"] / 20000, 1.0) * 0.15
        return depth_part + near_part + area_part

    return max(candidates, key=score)


# =====================================================
# YOLO
# =====================================================
def run_yolo(frame):
    h, w, _ = frame.shape
    results = yolo(frame, imgsz=YOLO_IMGSZ, verbose=False)

    yolo_boxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]
            conf = float(box.conf[0])

            if conf < 0.35:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w - 1, int(x2))
            y2 = min(h - 1, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            risk = get_object_risk(label)
            x_center = (x1 + x2) / 2
            direction = get_direction(x_center, w)

            yolo_boxes.append({
                "label": label,
                "conf": conf,
                "risk": risk,
                "direction": direction,
                "box": (x1, y1, x2, y2)
            })

    return yolo_boxes


def calc_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_yolo_with_depth_obstacle(yolo_boxes, depth_obstacle):
    if not depth_obstacle["detected"]:
        return None

    depth_box = depth_obstacle["box"]

    best_match = None
    best_iou = 0.0

    for ybox in yolo_boxes:
        iou = calc_iou(depth_box, ybox["box"])

        if iou > best_iou:
            best_iou = iou
            best_match = ybox

    if best_match and best_iou >= 0.1:
        return {
            "label": best_match["label"],
            "conf": best_match["conf"],
            "risk": best_match["risk"],
            "direction": depth_obstacle["direction"],
            "depth_score": depth_obstacle["depth_score"],
            "near_ratio": depth_obstacle["near_ratio"],
            "area": depth_obstacle["area"],
            "box": depth_box,
            "matched": True,
            "iou": best_iou
        }

    return {
        "label": "unknown obstacle",
        "conf": 0.0,
        "risk": "UNKNOWN",
        "direction": depth_obstacle["direction"],
        "depth_score": depth_obstacle["depth_score"],
        "near_ratio": depth_obstacle["near_ratio"],
        "area": depth_obstacle["area"],
        "box": depth_box,
        "matched": False,
        "iou": 0.0
    }


# =====================================================
# 진동 판단
# =====================================================
def should_vibrate(risk_result, confirm_count):
    """
    카메라 기반 진동 여부 결정.

    정책:
    1. unknown obstacle은 표시만 하고 진동하지 않음
    2. person/tree는 표시만 하고 진동하지 않음
    3. car/bicycle/kickboard는 CENTER일 때만 진동
    4. bollard/utility pole은 CENTER일 때 진동, 좌우는 매우 보수적으로
    5. LEFT/RIGHT 객체는 사용자의 진행 방향을 막는다고 보기 어려우므로 기본적으로 진동 억제
    """

    if risk_result is None:
        return False, None

    label = risk_result["label"]
    direction = risk_result["direction"]
    matched = risk_result["matched"]

    # YOLO와 매칭되지 않은 MiDaS unknown obstacle은 표시만
    if not matched:
        return False, None

    # 사람/나무는 진동하지 않고 표시만
    if label in NOTICE_OBJECTS:
        return False, None

    # 차량/자전거/킥보드
    if label in DYNAMIC_DANGER_OBJECTS:
        if direction == "CENTER":
            if confirm_count >= CENTER_DYNAMIC_CONFIRM:
                return True, "DOUBLE"
            return False, None

        # 좌우측 차량/자전거/킥보드는 표시만
        return False, None

    # 볼라드/전봇대 같은 고정 장애물
    if label in STATIC_OBSTACLES:
        if direction == "CENTER":
            if confirm_count >= CENTER_STATIC_CONFIRM:
                return True, "SHORT"
            return False, None

        # 좌우측 고정 장애물은 오래 연속 감지될 때만 약하게 진동
        if confirm_count >= SIDE_STATIC_CONFIRM:
            return True, "SHORT"

        return False, None

    return False, None
# =====================================================
# 시각화
# =====================================================
def draw_result(frame, risk_result, yolo_boxes, confirm_count):
    for ybox in yolo_boxes:
        x1, y1, x2, y2 = ybox["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 80), 1)
        cv2.putText(
            frame,
            f"{ybox['label']} {ybox['conf']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 180, 80),
            1
        )

    if risk_result is None:
        return frame

    x1, y1, x2, y2 = risk_result["box"]

    if risk_result["matched"]:
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    text = (
        f"{risk_result['label']} | {risk_result['direction']} | "
        f"d:{risk_result['depth_score']:.2f} r:{risk_result['near_ratio']:.2f} "
        f"c:{confirm_count}"
    )

    cv2.putText(
        frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    return frame


# =====================================================
# 메인
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

frame_count = 0
depth_map = None

print("MiDaS 중심 위험판단 V3 시작")
print("정책:")
print("- unknown obstacle: 표시만, 진동 없음")
print("- car/bicycle/kickboard/bollard/utility pole: 가까울 때 진동")
print("- person/tree: 표시만")
print("종료: q")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        h, w, _ = frame.shape

        if frame_count % MIDAS_INTERVAL == 0 or depth_map is None:
            depth_map = estimate_depth(frame)

        depth_obstacle = find_near_obstacle_from_depth(depth_map, w, h)
        yolo_boxes = run_yolo(frame)
        risk_result = match_yolo_with_depth_obstacle(yolo_boxes, depth_obstacle)

        if risk_result:
            current_direction = risk_result["direction"]
            current_label = risk_result["label"]

            if last_near_direction == current_direction and last_near_label == current_label:
                near_confirm_count += 1
            else:
                near_confirm_count = 1
                last_near_direction = current_direction
                last_near_label = current_label

            do_vibrate, pattern = should_vibrate(risk_result, near_confirm_count)

            print(
                f"[위험판단] 객체:{risk_result['label']} | "
                f"방향:{risk_result['direction']} | "
                f"depth:{risk_result['depth_score']:.2f} | "
                f"near_ratio:{risk_result['near_ratio']:.2f} | "
                f"matched:{risk_result['matched']} | "
                f"연속:{near_confirm_count} | "
                f"진동:{do_vibrate}"
            )

            if do_vibrate:
                trigger_vibration(risk_result["direction"], pattern=pattern)

        else:
            near_confirm_count = 0
            last_near_direction = None
            last_near_label = None
            all_vibrations_off()

        frame = draw_result(frame, risk_result, yolo_boxes, near_confirm_count)

        cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
        cv2.line(frame, (w * 2 // 3, 0), (w * 2 // 3, h), (255, 255, 255), 1)

        cv2.imshow("MiDaS First Vest V3", frame)

        if depth_map is not None:
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imshow("MiDaS Depth", depth_vis)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n종료합니다.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    all_vibrations_off()
    GPIO.cleanup()
