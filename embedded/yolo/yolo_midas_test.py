import cv2
import torch
import numpy as np
from ultralytics import YOLO

# =========================
# 설정
# =========================
YOLO_MODEL_PATH = "best.pt"
CAMERA_INDEX = 0
YOLO_IMGSZ = 320

# MiDaS는 CPU에서 무거우므로 매 프레임 돌리지 않고 N프레임마다 실행
MIDAS_INTERVAL = 5

# =========================
# 장치 설정
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# =========================
# YOLO 로드
# =========================
yolo = YOLO(YOLO_MODEL_PATH)
print("YOLO 클래스:", yolo.names)

# =========================
# MiDaS 로드
# =========================
print("MiDaS 로딩 중...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("MiDaS 로드 완료")

# =========================
# 위험도 기준
# =========================
HIGH_RISK = ["car", "bicycle", "kickboard"]
MEDIUM_RISK = ["person", "bollard", "tree", "utility pole"]


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


def normalize_depth(depth):
    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth)

    return (depth - depth_min) / (depth_max - depth_min)


def get_depth_level(depth_value):
    """
    MiDaS는 절대 cm가 아니라 상대 깊이.
    일반적으로 정규화된 값이 클수록 가까운 물체로 취급.
    """
    if depth_value >= 0.65:
        return "NEAR"
    elif depth_value >= 0.35:
        return "MID"
    else:
        return "FAR"


def estimate_depth(frame):
    """
    frame: BGR OpenCV 이미지
    return: 정규화된 depth map, shape = frame 크기
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_norm = normalize_depth(depth)

    return depth_norm


# =========================
# 카메라 열기
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

frame_count = 0
depth_map = None

print("YOLO + MiDaS 테스트 시작")
print("종료: q 키")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        h, w, _ = frame.shape

        # MiDaS는 N프레임마다 한 번만 실행
        if frame_count % MIDAS_INTERVAL == 0 or depth_map is None:
            depth_map = estimate_depth(frame)

        # YOLO 실행
        results = yolo(frame, imgsz=YOLO_IMGSZ, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = yolo.names[cls_id]
                conf = float(box.conf[0])

                if conf < 0.4:
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
                risk = get_object_risk(label)

                # bbox 영역 안의 MiDaS 상대 깊이 평균
                bbox_depth = depth_map[y1_i:y2_i, x1_i:x2_i]
                depth_value = float(np.mean(bbox_depth))
                depth_level = get_depth_level(depth_value)

                print(
                    f"객체: {label} | conf: {conf:.2f} | 방향: {direction} | "
                    f"위험도: {risk} | 상대깊이: {depth_value:.2f} | 거리판단: {depth_level}"
                )

                # 표시 색상
                if depth_level == "NEAR":
                    color = (0, 0, 255)
                elif depth_level == "MID":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)

                text = f"{label} {direction} {risk} {depth_level}"
                cv2.putText(
                    frame,
                    text,
                    (x1_i, max(20, y1_i - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        cv2.imshow("YOLO + MiDaS", frame)

        # depth map 시각화
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
