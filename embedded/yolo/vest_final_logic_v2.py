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

# =====================================================
# MiDaS 설정
# =====================================================
NEAR_THRESHOLD = 0.72
MIN_OBSTACLE_AREA = 1800
MAX_OBSTACLE_AREA_RATIO = 0.28
IGNORE_BOTTOM_RATIO = 0.82

MIN_BOX_W = 25
MIN_BOX_H = 35
MAX_BOX_WIDTH_RATIO = 0.75

# =====================================================
# 초음파 설정
# =====================================================
ULTRA_DANGER_DIST = 100.0       # 100cm 이하: 초음파 위험
ULTRA_EMERGENCY_DIST = 50.0     # 50cm 이하: 강한 진동
SIDE_TOLERANCE = 25.0

# 센서 실패 보정
MAX_FAIL_HOLD = 2

last_left_valid = None
last_right_valid = None
left_fail_count = 0
right_fail_count = 0

# =====================================================
# 카메라 위험 정책
# =====================================================
DYNAMIC_DANGER_OBJECTS = ["car", "bicycle", "kickboard"]
STATIC_OBSTACLES = ["bollard", "utility pole"]
NOTICE_OBJECTS = ["person", "tree"]

CENTER_DYNAMIC_CONFIRM = 2
CENTER_STATIC_CONFIRM = 2
SIDE_STATIC_CONFIRM = 5

near_confirm_count = 0
last_near_direction = None
last_near_label = None

# =====================================================
# 진동 설정
# =====================================================
VIB_CAMERA_COOLDOWN = 1.8
VIB_ULTRA_COOLDOWN = 0.4

last_camera_vib_time = 0
last_ultra_vib_time = 0

# =====================================================
# GPIO 설정
# =====================================================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

TRIG_LEFT, ECHO_LEFT = 11, 16
TRIG_RIGHT, ECHO_RIGHT = 12, 18

VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)
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
# 공통 함수
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


def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "LEFT"
    elif x_center < frame_width * 2 / 3:
        return "CENTER"
    else:
        return "RIGHT"


def get_object_risk(label):
    if label in DYNAMIC_DANGER_OBJECTS:
        return "HIGH"
    if label in STATIC_OBSTACLES or label in NOTICE_OBJECTS:
        return "MEDIUM"
    return "LOW"


def trigger_vibration(direction, pattern="SHORT", source="CAMERA"):
    global last_camera_vib_time, last_ultra_vib_time

    now = time.time()

    if source == "ULTRA":
        if now - last_ultra_vib_time < VIB_ULTRA_COOLDOWN:
            return
    else:
        if now - last_camera_vib_time < VIB_CAMERA_COOLDOWN:
            return

    pin = direction_to_pin(direction)
    if pin is None:
        return

    all_vibrations_off()

    if pattern == "STRONG":
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.4)
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

    if source == "ULTRA":
        last_ultra_vib_time = time.time()
    else:
        last_camera_vib_time = time.time()


# =====================================================
# 초음파 함수
# =====================================================
def get_distance_safe(trig, echo):
    GPIO.output(trig, False)
    time.sleep(0.000002)

    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    timeout = 0.02

    start_time = time.time()
    pulse_start = time.time()

    while GPIO.input(echo) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > timeout:
            return -1

    start_time = time.time()
    pulse_end = time.time()

    while GPIO.input(echo) == 1:
        pulse_end = time.time()
        if pulse_end - start_time > timeout:
            return -1

    duration = pulse_end - pulse_start
    distance = duration * 17150

    return round(distance, 1)


def stabilize_ultrasonic(raw_left, raw_right):
    global last_left_valid, last_right_valid
    global left_fail_count, right_fail_count

    if raw_left > 0:
        last_left_valid = raw_left
        left_fail_count = 0
        left = raw_left
    else:
        left_fail_count += 1
        if last_left_valid is not None and left_fail_count <= MAX_FAIL_HOLD:
            left = last_left_valid
        else:
            left = -1

    if raw_right > 0:
        last_right_valid = raw_right
        right_fail_count = 0
        right = raw_right
    else:
        right_fail_count += 1
        if last_right_valid is not None and right_fail_count <= MAX_FAIL_HOLD:
            right = last_right_valid
        else:
            right = -1

    return left, right


def judge_ultrasonic(left_dist, right_dist):
    left_valid = left_dist > 0
    right_valid = right_dist > 0

    left_danger = left_valid and left_dist <= ULTRA_DANGER_DIST
    right_danger = right_valid and right_dist <= ULTRA_DANGER_DIST

    if left_danger and right_danger:
        if abs(left_dist - right_dist) <= SIDE_TOLERANCE:
            return True, "CENTER", min(left_dist, right_dist)
        elif left_dist < right_dist:
            return True, "LEFT", left_dist
        else:
            return True, "RIGHT", right_dist

    if left_danger:
        return True, "LEFT", left_dist

    if right_danger:
        return True, "RIGHT", right_dist

    return False, None, None


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
    valid_depth = depth_map.copy()

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
        return None

    def score(c):
        depth_part = c["depth_score"] * 0.55
        near_part = c["near_ratio"] * 0.30
        area_part = min(c["area"] / 20000, 1.0) * 0.15
        return depth_part + near_part + area_part

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

            x_center = (x1 + x2) / 2
            direction = get_direction(x_center, w)

            yolo_boxes.append({
                "label": label,
                "conf": conf,
                "risk": get_object_risk(label),
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
    if depth_obstacle is None:
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
# 카메라 기반 진동 판단
# =====================================================
def update_camera_confirm(risk_result):
    global near_confirm_count, last_near_direction, last_near_label

    if risk_result is None:
        near_confirm_count = 0
        last_near_direction = None
        last_near_label = None
        return 0

    current_direction = risk_result["direction"]
    current_label = risk_result["label"]

    if last_near_direction == current_direction and last_near_label == current_label:
        near_confirm_count += 1
    else:
        near_confirm_count = 1
        last_near_direction = current_direction
        last_near_label = current_label

    return near_confirm_count


def should_camera_vibrate(risk_result, confirm_count):
    if risk_result is None:
        return False, None

    if not risk_result["matched"]:
        return False, None

    label = risk_result["label"]
    direction = risk_result["direction"]

    if label in NOTICE_OBJECTS:
        return False, None

    if label in DYNAMIC_DANGER_OBJECTS:
        if direction == "CENTER" and confirm_count >= CENTER_DYNAMIC_CONFIRM:
            return True, "DOUBLE"
        return False, None

    if label in STATIC_OBSTACLES:
        if direction == "CENTER" and confirm_count >= CENTER_STATIC_CONFIRM:
            return True, "SHORT"
        if direction != "CENTER" and confirm_count >= SIDE_STATIC_CONFIRM:
            return True, "SHORT"
        return False, None

    return False, None


def make_message_candidate(source, direction, label=None, distance=None):
    direction_kr = {
        "LEFT": "왼쪽",
        "CENTER": "전방",
        "RIGHT": "오른쪽"
    }.get(direction, "전방")

    label_kr = {
        "car": "차량",
        "bicycle": "자전거",
        "kickboard": "킥보드",
        "person": "사람",
        "bollard": "볼라드",
        "tree": "나무",
        "utility pole": "전봇대",
        "unknown obstacle": "장애물"
    }.get(label, "장애물")

    if source == "ULTRA":
        return f"{direction_kr}에 가까운 장애물이 있습니다."

    return f"{direction_kr}에 {label_kr}이 감지되었습니다."


# =====================================================
# 시각화
# =====================================================
def draw_result(frame, risk_result, yolo_boxes, ultra_info=None, confirm_count=0):
    h, w, _ = frame.shape

    cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
    cv2.line(frame, (w * 2 // 3, 0), (w * 2 // 3, h), (255, 255, 255), 1)

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

    if risk_result is not None:
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

    if ultra_info is not None:
        raw_left, raw_right, left_dist, right_dist, detected, direction, distance = ultra_info

        ultra_text = f"ULTRA L:{left_dist}cm R:{right_dist}cm"

        if detected:
            ultra_text += f" | DANGER {direction} {distance}cm"
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)

        cv2.putText(
            frame,
            ultra_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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
prev_time = time.time()

print("Vest Final Logic V2 시작")
print("정책:")
print("- 1순위: 초음파 50cm 이하 STRONG, 50~100cm SHORT")
print("- 2순위: 카메라 CENTER 위험 객체만 보수적으로 진동")
print("- unknown/person/tree는 표시 및 메시지 후보만")
print("종료: q")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        h, w, _ = frame.shape

        # -----------------------------
        # 1. 초음파 측정 및 안정화
        # -----------------------------
        raw_left = get_distance_safe(TRIG_LEFT, ECHO_LEFT)
        time.sleep(0.02)
        raw_right = get_distance_safe(TRIG_RIGHT, ECHO_RIGHT)

        left_dist, right_dist = stabilize_ultrasonic(raw_left, raw_right)
        ultra_detected, ultra_direction, ultra_distance = judge_ultrasonic(left_dist, right_dist)

        ultra_info = (
            raw_left,
            raw_right,
            left_dist,
            right_dist,
            ultra_detected,
            ultra_direction,
            ultra_distance
        )

        # -----------------------------
        # 2. 초음파 긴급 우선
        # -----------------------------
        if ultra_detected:
            pattern = "STRONG" if ultra_distance <= ULTRA_EMERGENCY_DIST else "SHORT"

            msg = make_message_candidate("ULTRA", ultra_direction, distance=ultra_distance)
            print(
                f"[초음파 긴급] L:{left_dist}cm R:{right_dist}cm | "
                f"방향:{ultra_direction} | 거리:{ultra_distance}cm | "
                f"패턴:{pattern} | 메시지:{msg}"
            )

            trigger_vibration(ultra_direction, pattern=pattern, source="ULTRA")

        # -----------------------------
        # 3. MiDaS 갱신
        # -----------------------------
        if frame_count % MIDAS_INTERVAL == 0 or depth_map is None:
            depth_map = estimate_depth(frame)

        # -----------------------------
        # 4. MiDaS 중심 + YOLO 보조
        # -----------------------------
        depth_obstacle = find_near_obstacle_from_depth(depth_map, w, h)
        yolo_boxes = run_yolo(frame)
        risk_result = match_yolo_with_depth_obstacle(yolo_boxes, depth_obstacle)

        confirm_count = update_camera_confirm(risk_result)
        do_camera_vib, camera_pattern = should_camera_vibrate(risk_result, confirm_count)

        if risk_result is not None:
            msg = make_message_candidate(
                "CAMERA",
                risk_result["direction"],
                label=risk_result["label"]
            )

            print(
                f"[카메라 판단] 객체:{risk_result['label']} | "
                f"방향:{risk_result['direction']} | "
                f"matched:{risk_result['matched']} | "
                f"depth:{risk_result['depth_score']:.2f} | "
                f"near_ratio:{risk_result['near_ratio']:.2f} | "
                f"연속:{confirm_count} | "
                f"진동:{do_camera_vib} | "
                f"메시지후보:{msg}"
            )

            if not ultra_detected and do_camera_vib:
                trigger_vibration(
                    risk_result["direction"],
                    pattern=camera_pattern,
                    source="CAMERA"
                )

        elif not ultra_detected:
            all_vibrations_off()

        # -----------------------------
        # 5. FPS 표시
        # -----------------------------
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS:{fps:.2f} | DEVICE:{device}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # -----------------------------
        # 6. 화면 표시
        # -----------------------------
        frame = draw_result(
            frame,
            risk_result,
            yolo_boxes,
            ultra_info=ultra_info,
            confirm_count=confirm_count
        )

        cv2.imshow("Vest Final Logic V2", frame)

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
