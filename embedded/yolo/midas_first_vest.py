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
MIDAS_INTERVAL = 5

# MiDaS 가까운 영역 기준
NEAR_THRESHOLD = 0.65       # 0~1 정규화 depth 중 가까운 영역 기준
MIN_OBSTACLE_AREA = 2500    # 너무 작은 노이즈 제거
CENTER_TOLERANCE = 0.12     # 화면 중앙 판단 보정

# =====================================================
# GPIO 설정
# =====================================================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

VIB_COOLDOWN = 0.45
last_vib_time = 0

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

    if pattern == "STRONG":
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.35)
        GPIO.output(pin, GPIO.LOW)

    elif pattern == "DOUBLE":
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
# 방향/위험도 함수
# =====================================================
def get_direction(x_center, frame_width):
    left_boundary = frame_width / 3
    right_boundary = frame_width * 2 / 3

    if x_center < left_boundary:
        return "LEFT"
    elif x_center < right_boundary:
        return "CENTER"
    else:
        return "RIGHT"

def get_object_risk(label):
    high = ["car", "bicycle", "kickboard"]
    medium = ["person", "bollard", "tree", "utility pole"]

    if label in high:
        return "HIGH"
    if label in medium:
        return "MEDIUM"
    return "LOW"

# =====================================================
# MiDaS 함수
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
    MiDaS depth map만 보고 가까운 장애물 후보를 찾는다.
    YOLO가 못 잡아도 가까운 영역이 있으면 unknown obstacle로 처리한다.
    """

    # 가까운 영역 마스크 생성
    near_mask = (depth_map >= NEAR_THRESHOLD).astype(np.uint8) * 255

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN, kernel)
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        near_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_OBSTACLE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 화면 맨 아래/가장자리의 과도한 노이즈 방지용
        if h < 20 or w < 20:
            continue

        x_center = x + w / 2
        direction = get_direction(x_center, frame_width)

        region = depth_map[y:y+h, x:x+w]
        depth_score = float(np.mean(region))

        candidates.append({
            "detected": True,
            "direction": direction,
            "box": (x, y, x + w, y + h),
            "area": area,
            "depth_score": depth_score
        })

    if not candidates:
        return {
            "detected": False,
            "direction": None,
            "box": None,
            "area": 0,
            "depth_score": 0
        }

    # 가까움 점수와 면적을 함께 고려
    def score(c):
        return c["depth_score"] * 0.7 + min(c["area"] / 30000, 1.0) * 0.3

    return max(candidates, key=score)

# =====================================================
# YOLO 함수
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
    """
    MiDaS가 찾은 가까운 영역과 YOLO 박스가 겹치면 객체명을 붙인다.
    안 겹치면 unknown obstacle.
    """

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
            "area": depth_obstacle["area"],
            "box": depth_box,
            "matched": True,
            "iou": best_iou
        }

    return {
        "label": "unknown obstacle",
        "conf": 0.0,
        "risk": "MEDIUM",
        "direction": depth_obstacle["direction"],
        "depth_score": depth_obstacle["depth_score"],
        "area": depth_obstacle["area"],
        "box": depth_box,
        "matched": False,
        "iou": 0.0
    }

# =====================================================
# 시각화
# =====================================================
def draw_result(frame, risk_result, yolo_boxes):
    # YOLO 박스는 얇게 표시
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

    # MiDaS 기반 위험 박스는 굵게 표시
    if risk_result["matched"]:
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    text = (
        f"{risk_result['label']} | {risk_result['direction']} | "
        f"depth:{risk_result['depth_score']:.2f}"
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

print("MiDaS 중심 위험판단 테스트 시작")
print("종료: q")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        h, w, _ = frame.shape

        # MiDaS 갱신
        if frame_count % MIDAS_INTERVAL == 0 or depth_map is None:
            depth_map = estimate_depth(frame)

        # MiDaS가 먼저 가까운 영역 탐지
        depth_obstacle = find_near_obstacle_from_depth(depth_map, w, h)

        # YOLO는 객체명 보조
        yolo_boxes = run_yolo(frame)

        risk_result = match_yolo_with_depth_obstacle(yolo_boxes, depth_obstacle)

        if risk_result:
            print(
                f"[MiDaS 중심] 객체: {risk_result['label']} | "
                f"방향: {risk_result['direction']} | "
                f"depth: {risk_result['depth_score']:.2f} | "
                f"matched: {risk_result['matched']}"
            )

            # MiDaS 기반으로 진동
            if risk_result["matched"] and risk_result["risk"] == "HIGH":
                trigger_vibration(risk_result["direction"], pattern="DOUBLE")
            else:
                trigger_vibration(risk_result["direction"], pattern="SHORT")

        else:
            all_vibrations_off()

        frame = draw_result(frame, risk_result, yolo_boxes)

        # 방향 구역 표시
        cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
        cv2.line(frame, (w * 2 // 3, 0), (w * 2 // 3, h), (255, 255, 255), 1)

        cv2.imshow("MiDaS First Vest", frame)

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
