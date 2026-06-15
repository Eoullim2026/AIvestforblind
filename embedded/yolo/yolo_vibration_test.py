from ultralytics import YOLO
import cv2
import Jetson.GPIO as GPIO
import time

# =========================
# GPIO 설정
# =========================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

# =========================
# YOLO 설정
# =========================
model = YOLO("best.pt")

HIGH_RISK = ["car", "bicycle", "kickboard"]
MEDIUM_RISK = ["person", "bollard", "tree", "utility pole"]

# 진동 유지 시간
VIB_DURATION = 0.15

# 너무 자주 울리지 않게 하는 쿨타임
VIB_COOLDOWN = 0.5
last_vibration_time = 0


def all_vibrations_off():
    GPIO.output([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.LOW)


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


def direction_to_pin(direction):
    if direction == "LEFT":
        return VIB_LEFT
    elif direction == "CENTER":
        return VIB_CENTER
    elif direction == "RIGHT":
        return VIB_RIGHT
    return None


def trigger_vibration(direction, risk):
    global last_vibration_time

    now = time.time()
    if now - last_vibration_time < VIB_COOLDOWN:
        return

    pin = direction_to_pin(direction)
    if pin is None:
        return

    all_vibrations_off()

    # HIGH는 조금 더 길게, MEDIUM은 짧게
    duration = 0.25 if risk == "HIGH" else VIB_DURATION

    GPIO.output(pin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)

    last_vibration_time = time.time()


def select_most_important_detection(detections):
    """
    여러 객체가 잡혔을 때 가장 중요한 객체 하나 선택
    우선순위:
    1. HIGH 위험 객체
    2. bbox 면적이 큰 객체
    """
    if not detections:
        return None

    def score(det):
        risk_score = 2 if det["risk"] == "HIGH" else 1
        return (risk_score, det["area"])

    return max(detections, key=score)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("🚀 YOLO 기반 진동 피드백 시작")
print("종료하려면 q 키를 누르세요.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break

        h, w, _ = frame.shape
        results = model(frame, imgsz=320, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # 너무 낮은 confidence는 무시
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2
                area = (x2 - x1) * (y2 - y1)

                direction = get_direction(x_center, w)
                risk = get_object_risk(label)

                if risk == "LOW":
                    continue

                detections.append({
                    "label": label,
                    "conf": conf,
                    "direction": direction,
                    "risk": risk,
                    "area": area,
                    "box": (x1, y1, x2, y2)
                })

                color = (0, 0, 255) if risk == "HIGH" else (0, 255, 255)

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} {direction} {risk} {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        target = select_most_important_detection(detections)

        if target:
            print(
                f"감지: {target['label']} | 방향: {target['direction']} | "
                f"위험도: {target['risk']} | conf: {target['conf']:.2f}"
            )
            trigger_vibration(target["direction"], target["risk"])
        else:
            all_vibrations_off()

        cv2.imshow("YOLO Vibration Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n종료합니다.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    all_vibrations_off()
    GPIO.cleanup()
