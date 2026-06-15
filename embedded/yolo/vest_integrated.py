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
MIDAS_INTERVAL = 8  # CPU 부담 줄이기 위해 N프레임마다 MiDaS 실행

# =====================================================
# GPIO 핀 설정
# =====================================================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 초음파 센서 핀
TRIG_LEFT, ECHO_LEFT = 11, 16
TRIG_RIGHT, ECHO_RIGHT = 12, 18

# 진동모듈 핀
VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)
GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

# =====================================================
# 위험 판단 기준
# =====================================================
ULTRASONIC_DANGER_DIST = 100.0  # cm 이하 즉시 진동
SIDE_TOLERANCE = 25.0

HIGH_RISK = ["car", "bicycle", "kickboard"]
MEDIUM_RISK = ["person", "bollard", "tree", "utility pole"]

# 진동 패턴 설정
VIB_COOLDOWN = 0.35
last_vib_time = 0

PATTERN_ULTRASONIC = "STRONG"
PATTERN_CAMERA_HIGH = "DOUBLE"
PATTERN_CAMERA_MEDIUM = "SHORT"

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


def trigger_vibration(direction, pattern="SHORT", force=False):
    """
    방향과 위험 상황에 따라 진동 패턴 제어

    SHORT  : 짧은 1회 진동
    DOUBLE : 짧게 2회 진동
    STRONG : 비교적 긴 1회 진동
    """

    global last_vib_time

    now = time.time()

    if not force and now - last_vib_time < VIB_COOLDOWN:
        return

    pin = direction_to_pin(direction)

    if pin is None:
        return

    all_vibrations_off()

    if pattern == "STRONG":
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.45)
        GPIO.output(pin, GPIO.LOW)

    elif pattern == "DOUBLE":
        for _ in range(2):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.15)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.08)

    else:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.15)
        GPIO.output(pin, GPIO.LOW)

    last_vib_time = time.time()


def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "LEFT"
    elif x_center < frame_width * 2 / 3:
        return "CENTER"
    else:
        return "RIGHT"


def get_object_risk(label):
    if label in HIGH_RISK:
        return "HIGH"
    elif label in MEDIUM_RISK:
        return "MEDIUM"
    return "LOW"


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


def judge_ultrasonic(left_dist, right_dist):
    left_valid = left_dist > 0
    right_valid = right_dist > 0

    left_danger = left_valid and left_dist <= ULTRASONIC_DANGER_DIST
    right_danger = right_valid and right_dist <= ULTRASONIC_DANGER_DIST

    if left_danger and right_danger:
        if abs(left_dist - right_dist) <= SIDE_TOLERANCE:
            return {
                "detected": True,
                "direction": "CENTER",
                "reason": "ULTRASONIC_CENTER",
                "distance": min(left_dist, right_dist),
            }
        elif left_dist < right_dist:
            return {
                "detected": True,
                "direction": "LEFT",
                "reason": "ULTRASONIC_LEFT",
                "distance": left_dist,
            }
        else:
            return {
                "detected": True,
                "direction": "RIGHT",
                "reason": "ULTRASONIC_RIGHT",
                "distance": right_dist,
            }

    if left_danger:
        return {
            "detected": True,
            "direction": "LEFT",
            "reason": "ULTRASONIC_LEFT",
            "distance": left_dist,
        }

    if right_danger:
        return {
            "detected": True,
            "direction": "RIGHT",
            "reason": "ULTRASONIC_RIGHT",
            "distance": right_dist,
        }

    return {
        "detected": False,
        "direction": None,
        "reason": "ULTRASONIC_SAFE",
        "distance": None,
    }


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


def get_depth_level(depth_value):
    # MiDaS 정규화 값 기준. 환경에 따라 조정 가능.
    if depth_value >= 0.65:
        return "NEAR"
    elif depth_value >= 0.35:
        return "MID"
    else:
        return "FAR"


# =====================================================
# YOLO + MiDaS 판단
# =====================================================
def select_best_camera_risk(frame, depth_map):
    h, w, _ = frame.shape
    results = yolo(frame, imgsz=YOLO_IMGSZ, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]
            conf = float(box.conf[0])

            if conf < 0.4:
                continue

            risk = get_object_risk(label)
            if risk == "LOW":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1_i = max(0, int(x1))
            y1_i = max(0, int(y1))
            x2_i = min(w - 1, int(x2))
            y2_i = min(h - 1, int(y2))

            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            x_center = (x1 + x2) / 2
            direction = get_direction(x_center, w)

            bbox_depth = depth_map[y1_i:y2_i, x1_i:x2_i]
            depth_value = float(np.mean(bbox_depth))
            depth_level = get_depth_level(depth_value)

            area = (x2 - x1) * (y2 - y1)

            detections.append({
                "label": label,
                "conf": conf,
                "risk": risk,
                "direction": direction,
                "depth_value": depth_value,
                "depth_level": depth_level,
                "area": area,
                "box": (x1_i, y1_i, x2_i, y2_i),
            })

    if not detections:
        return None, results

    def score(det):
        risk_score = 2 if det["risk"] == "HIGH" else 1
        depth_score = 3 if det["depth_level"] == "NEAR" else 2 if det["depth_level"] == "MID" else 1
        return (depth_score, risk_score, det["area"])

    return max(detections, key=score), results


def draw_detection(frame, target):
    if target is None:
        return frame

    x1, y1, x2, y2 = target["box"]

    if target["depth_level"] == "NEAR":
        color = (0, 0, 255)
    elif target["depth_level"] == "MID":
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = (
        f"{target['label']} {target['direction']} "
        f"{target['risk']} {target['depth_level']}"
    )

    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    return frame


# =====================================================
# 메인 루프
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

frame_count = 0
depth_map = None

print("통합 시스템 시작")
print("종료하려면 q 키")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        # -----------------------------
        # 1. 초음파 측정
        # -----------------------------
        left_dist = get_distance_safe(TRIG_LEFT, ECHO_LEFT)
        time.sleep(0.02)
        right_dist = get_distance_safe(TRIG_RIGHT, ECHO_RIGHT)

        ultrasonic_risk = judge_ultrasonic(left_dist, right_dist)

        print(f"[초음파] LEFT: {left_dist}cm | RIGHT: {right_dist}cm")

        # -----------------------------
        # 2. 초음파 긴급 우선 처리
        # -----------------------------
        if ultrasonic_risk["detected"]:
            direction = ultrasonic_risk["direction"]
            distance = ultrasonic_risk["distance"]

            print(f"🚨 [초음파 긴급] 방향: {direction} | 거리: {distance}cm")
            trigger_vibration(direction, pattern=PATTERN_ULTRASONIC, force=True)

            cv2.putText(
                frame,
                f"ULTRASONIC {direction} {distance}cm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # -----------------------------
        # 3. MiDaS 갱신
        # -----------------------------
        if frame_count % MIDAS_INTERVAL == 0 or depth_map is None:
            depth_map = estimate_depth(frame)

        # -----------------------------
        # 4. YOLO + MiDaS 위험 판단
        # -----------------------------
        target, _ = select_best_camera_risk(frame, depth_map)

        if target:
            print(
                f"[카메라] 객체: {target['label']} | 방향: {target['direction']} | "
                f"위험도: {target['risk']} | 상대깊이: {target['depth_value']:.2f} | "
                f"거리판단: {target['depth_level']}"
            )

            # 초음파 긴급이 없을 때만 카메라 기반 진동
            if not ultrasonic_risk["detected"] and target["depth_level"] == "NEAR":
                if target["risk"] == "HIGH":
                    pattern = PATTERN_CAMERA_HIGH
                else:
                    pattern = PATTERN_CAMERA_MEDIUM

                print(f"⚠️ [카메라 위험] {target['direction']} 진동 | 패턴: {pattern}")
                trigger_vibration(target["direction"], pattern=pattern)

            frame = draw_detection(frame, target)

        else:
            if not ultrasonic_risk["detected"]:
                all_vibrations_off()

        # -----------------------------
        # 5. 화면 표시
        # -----------------------------
        cv2.imshow("Vest Integrated", frame)

        if depth_map is not None:
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imshow("MiDaS Depth", depth_vis)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n시스템 종료")

finally:
    cap.release()
    cv2.destroyAllWindows()
    all_vibrations_off()
    GPIO.cleanup()
