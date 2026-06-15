from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

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
    elif label in ["person", "bollard", "tree", "utility pole"]:
        return "MEDIUM"
    return "LOW"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    h, w, _ = frame.shape

    results = model(frame, imgsz=320, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2

            direction = get_direction(x_center, w)
            risk = get_object_risk(label)

            print(f"객체: {label} | 방향: {direction} | 위험도: {risk}")

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {direction} {risk}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Risk Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
