import cv2
import time
import torch
import numpy as np
import Jetson.GPIO as GPIO
from ultralytics import YOLO

# TTS / audio output
import subprocess
import threading
import asyncio
import json
import queue
import signal
from gtts import gTTS
import websockets
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import urllib.request
import urllib.error
import re


# =====================================================
# 기본 설정
# =====================================================
YOLO_MODEL_PATH = "best.pt"
CAMERA_INDEX = 0

# FPS 개선 설정
YOLO_IMGSZ = 320
MIDAS_INTERVAL = 3          # MiDaS를 3프레임마다 실행
ULTRA_INTERVAL = 3          # 초음파를 3프레임마다 측정
SHOW_DEPTH_WINDOW = False   # depth 창은 FPS를 깎으므로 기본 OFF
DEBUG_PRINT = False         # Jetson에서 print도 FPS 저하 요인이라 기본 OFF
DRAW_GUIDE_LINES = False    # 개발용 좌/중/우 분할선 표시 여부

# 카메라 입력 해상도 설정
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


# =====================================================
# MediaMTX RTSP 송출 설정
# =====================================================
STREAM_ENABLE = True
STREAM_URL = "rtsp://220.69.21.83:8554/vest"

# 송출 해상도/FPS
# 객체인식 + MiDaS + H.264 소프트웨어 인코딩을 같이 돌리므로 낮게 시작
STREAM_WIDTH = 854
STREAM_HEIGHT = 480
STREAM_FPS = 10


# =====================================================
# WebSocket 이벤트 로그 설정
# =====================================================
EVENT_LOG_ENABLE = True
EVENT_WS_URL = "ws://220.69.21.83:8765"

# 5초마다 상태 로그 전송
PERIODIC_EVENT_INTERVAL = 5.0

# 실제 진동이 울렸을 때 남기는 로그의 최소 간격
VIBRATION_EVENT_COOLDOWN = 1.0

last_periodic_event_time = 0
last_vibration_event_time = {}


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

# 초음파 이전값 재사용용
last_raw_left = -1
last_raw_right = -1
last_left_dist = -1
last_right_dist = -1
last_ultra_detected = False
last_ultra_direction = None
last_ultra_distance = None


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

# 진동 스레드 중복 방지용
vibration_lock = threading.Lock()
vibration_busy = False


# =====================================================
# TTS 설정
# =====================================================
TTS_COOLDOWN = 3.0
TTS_REPEAT_COOLDOWN = 6.0
TTS_PATH = "/tmp/vest_tts.mp3"

last_tts_time = 0
last_tts_message = None


# =====================================================
# 화면 해설 캡셔닝 설정
# =====================================================
# 위험 감지 루프를 막지 않도록 별도 worker에서 정해진 주기마다 최신 프레임 1장만 처리한다.
SCENE_CAPTION_ENABLE = True
SCENE_CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
SCENE_CAPTION_INTERVAL = 15.0
SCENE_CAPTION_TTS_ENABLE = True
SCENE_CAPTION_EVENT_ENABLE = True
SCENE_TTS_PATH = "/tmp/vest_scene_tts.mp3"

# 위험/일반 TTS 직후에는 화면 해설이 끼어들지 않게 억제한다.
SCENE_TTS_SUPPRESS_AFTER_ALERT = 5.0
SCENE_PRINT_DEBUG = True

# =====================================================
# Qwen 화면 해설 변환 설정
# =====================================================
# Qwen은 Python에서 직접 로드하지 않는다.
# 별도 터미널에서 llama.cpp llama-server를 먼저 실행한 뒤 localhost API로 호출한다.
# 실행 예:
# ./build/bin/llama-server \
#   -m ~/capstone2026/yolo/models/qwen-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf \
#   -c 1024 -ngl 99 --host 127.0.0.1 --port 8080
QWEN_SCENE_ENABLE = True
QWEN_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
QWEN_API_TIMEOUT = 5.0
QWEN_MAX_TOKENS = 40
QWEN_TEMPERATURE = 0
QWEN_PRINT_DEBUG = True

scene_caption_queue = queue.Queue(maxsize=1)
scene_tts_queue = queue.Queue(maxsize=1)
scene_caption_running = False
scene_tts_running = False
scene_caption_stop = False
last_scene_caption_submit_time = 0

scene_caption_processor = None
scene_caption_model = None
scene_caption_device = None

qwen_available = QWEN_SCENE_ENABLE

# 종료 제어 플래그
shutdown_requested = False


def request_shutdown(signum=None, frame=None):
    global shutdown_requested, scene_caption_stop
    shutdown_requested = True
    scene_caption_stop = True
    print("\n[SHUTDOWN] 종료 신호를 받았습니다. 정리 중...")


signal.signal(signal.SIGINT, request_shutdown)
signal.signal(signal.SIGTERM, request_shutdown)


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


def load_scene_caption_model():
    """BLIP 이미지 캡셔닝 모델을 로드한다. 실패하면 화면 해설만 비활성화한다."""
    if not SCENE_CAPTION_ENABLE:
        return None, None, None

    caption_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print("BLIP 화면 해설 모델 로딩 중...")
        processor = BlipProcessor.from_pretrained(SCENE_CAPTION_MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(
            SCENE_CAPTION_MODEL_NAME,
            use_safetensors=False
        )
        model.to(caption_device)
        model.eval()
        print(f"BLIP 화면 해설 모델 로드 완료: {caption_device}")
        return processor, model, caption_device

    except Exception as e:
        print(f"[SCENE] BLIP 모델 로드 실패. 화면 해설 비활성화: {e}")
        return None, None, None


scene_caption_processor, scene_caption_model, scene_caption_device = load_scene_caption_model()


def check_qwen_scene_api():
    """Qwen llama-server는 외부 프로세스로 실행된다. 여기서는 사용 설정만 확인한다."""
    if not QWEN_SCENE_ENABLE:
        print("[QWEN API] Qwen 화면 해설 변환 비활성화")
        return False

    print(f"[QWEN API] localhost Qwen 서버 사용: {QWEN_API_URL}")
    print("[QWEN API] llama-server가 꺼져 있으면 영어 캡션으로 fallback합니다.")
    return True


qwen_available = check_qwen_scene_api()


# =====================================================
# 공통 함수
# =====================================================
def debug_log(message):
    if DEBUG_PRINT:
        print(message)


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


# =====================================================
# WebSocket 이벤트 로그 함수
# =====================================================
class EventSender:
    """
    Jetson에서 발생한 감지/진동 이벤트를 학교 서버 WebSocket 중계 서버로 전송한다.
    - 영상: RTSP로 MediaMTX에 송출
    - 로그: WebSocket으로 event_ws_server.py에 전송
    """

    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.loop = None
        self.queue = None
        self.thread = None
        self.running = False

    def start(self):
        if not EVENT_LOG_ENABLE:
            return

        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

        if self.loop and self.queue:
            try:
                asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)
            except Exception:
                pass

    def send_event(self, event):
        if not EVENT_LOG_ENABLE:
            return

        if not self.running or self.loop is None or self.queue is None:
            return

        try:
            asyncio.run_coroutine_threadsafe(self.queue.put(event), self.loop)
        except Exception as e:
            print(f"[EVENT 큐 오류] {e}")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.queue = asyncio.Queue()
        self.loop.run_until_complete(self._sender_loop())

    async def _drain_messages(self, ws):
        """중계 서버가 broadcast한 메시지를 읽어서 수신 버퍼가 쌓이지 않게 비운다."""
        try:
            async for _ in ws:
                pass
        except Exception:
            pass

    async def _sender_loop(self):
        while self.running:
            try:
                print(f"[EVENT] WebSocket 연결 시도: {self.ws_url}")

                async with websockets.connect(self.ws_url) as ws:
                    print("[EVENT] WebSocket 연결됨")
                    drain_task = asyncio.create_task(self._drain_messages(ws))

                    while self.running:
                        event = await self.queue.get()

                        if event is None:
                            break

                        payload = json.dumps(event, ensure_ascii=False)
                        await ws.send(payload)
                        debug_log(f"[EVENT] 전송: {payload}")

                    drain_task.cancel()

            except Exception as e:
                print(f"[EVENT] WebSocket 오류: {e}")
                await asyncio.sleep(3)


def direction_to_korean(direction):
    return {
        "LEFT": "좌측",
        "CENTER": "중앙",
        "RIGHT": "우측"
    }.get(direction, "중앙")


def label_to_korean(label):
    return {
        "car": "차량",
        "bicycle": "자전거",
        "kickboard": "킥보드",
        "person": "사람",
        "bollard": "볼라드",
        "tree": "나무",
        "utility pole": "전봇대",
        "unknown obstacle": "장애물"
    }.get(label, label if label else "감지 객체 없음")


def pattern_to_korean(pattern):
    return {
        "STRONG": "강한 진동",
        "DOUBLE": "이중 진동",
        "SHORT": "짧은 진동"
    }.get(pattern, "진동")


def make_event_payload(
    name,
    direction,
    distance=None,
    confidence=None,
    level="INFO",
    message="",
    source="CAMERA",
    vibration=False,
    pattern=None
):
    direction_kr = direction_to_korean(direction)

    if isinstance(distance, (int, float)) and distance > 0:
        if source == "ULTRA":
            distance_text = f"{distance / 100.0:.1f}m"
        else:
            distance_text = f"{distance:.1f}m"
    elif isinstance(distance, str):
        distance_text = distance
    else:
        distance_text = "-"

    if isinstance(confidence, (int, float)):
        confidence_text = f"{confidence * 100:.0f}%"
    elif isinstance(confidence, str):
        confidence_text = confidence
    else:
        confidence_text = "-"

    return {
        "name": name,
        "direction": direction_kr,
        "distance": distance_text,
        "confidence": confidence_text,
        "level": level,
        "message": message,
        "source": source,
        "vibration": vibration,
        "pattern": pattern_to_korean(pattern) if pattern else "-",
        "timestamp": time.strftime("%H:%M:%S"),
    }


def should_send_periodic_event():
    global last_periodic_event_time

    now = time.time()

    if now - last_periodic_event_time >= PERIODIC_EVENT_INTERVAL:
        last_periodic_event_time = now
        return True

    return False


def should_send_vibration_event(source, direction, name, pattern):
    key = f"{source}:{direction}:{name}:{pattern}"
    now = time.time()
    last_time = last_vibration_event_time.get(key, 0)

    if now - last_time >= VIBRATION_EVENT_COOLDOWN:
        last_vibration_event_time[key] = now
        return True

    return False


def send_periodic_status_event(event_sender, ultra_detected, ultra_direction, ultra_distance, risk_result):
    """5초마다 현재 상태를 1건 전송한다. 초음파 위험이 있으면 초음파를 우선 기록한다."""

    if not should_send_periodic_event():
        return

    if ultra_detected:
        level = "HIGH" if ultra_distance <= ULTRA_EMERGENCY_DIST else "MEDIUM"
        event = make_event_payload(
            name="초음파 장애물",
            direction=ultra_direction,
            distance=ultra_distance,
            confidence="-",
            level=level,
            message="주기 상태 로그: 초음파 장애물 감지",
            source="ULTRA",
            vibration=False,
        )
        event_sender.send_event(event)
        return

    if risk_result is not None:
        label = risk_result.get("label", "unknown obstacle")
        name = label_to_korean(label)
        level = risk_result.get("risk", "INFO")

        if level == "UNKNOWN":
            level = "MEDIUM"

        event = make_event_payload(
            name=name,
            direction=risk_result.get("direction"),
            distance=None,
            confidence=risk_result.get("conf"),
            level=level,
            message="주기 상태 로그: 카메라 객체 감지",
            source="CAMERA",
            vibration=False,
        )
        event_sender.send_event(event)
        return

    event = make_event_payload(
        name="감지 객체 없음",
        direction="CENTER",
        distance="-",
        confidence="-",
        level="INFO",
        message="주기 상태 로그: 감지 객체 없음",
        source="SYSTEM",
        vibration=False,
    )
    event_sender.send_event(event)


def send_vibration_log_event(event_sender, source, direction, name, distance, confidence, level, pattern):
    if not should_send_vibration_event(source, direction, name, pattern):
        return

    pattern_kr = pattern_to_korean(pattern)

    event = make_event_payload(
        name=name,
        direction=direction,
        distance=distance,
        confidence=confidence,
        level=level,
        message=f"진동 피드백 발생: {pattern_kr}",
        source=source,
        vibration=True,
        pattern=pattern,
    )

    event_sender.send_event(event)


# =====================================================
# FFmpeg RTSP 송출 함수
# =====================================================
def start_ffmpeg_stream(width, height, fps, url):
    """
    OpenCV frame(BGR)을 FFmpeg stdin으로 전달하고,
    FFmpeg가 H.264로 인코딩한 뒤 MediaMTX로 RTSP 송출한다.
    """
    command = [
        "ffmpeg",
        "-y",

        # stdin으로 raw BGR frame 입력
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",

        # 오디오 없음
        "-an",

        # H.264 소프트웨어 인코딩
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",

        # 낮은 비트레이트로 시작
        "-b:v", "800k",
        "-maxrate", "800k",
        "-bufsize", "1600k",

        # RTSP 송출
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        url
    ]

    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# =====================================================
# 비동기 진동 함수
# =====================================================
def _vibration_worker(pin, pattern):
    global vibration_busy

    try:
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

    finally:
        all_vibrations_off()
        with vibration_lock:
            vibration_busy = False


def trigger_vibration(direction, pattern="SHORT", source="CAMERA"):
    global last_camera_vib_time, last_ultra_vib_time, vibration_busy

    now = time.time()

    if source == "ULTRA":
        if now - last_ultra_vib_time < VIB_ULTRA_COOLDOWN:
            return False
    else:
        if now - last_camera_vib_time < VIB_CAMERA_COOLDOWN:
            return False

    pin = direction_to_pin(direction)
    if pin is None:
        return False

    with vibration_lock:
        if vibration_busy:
            return False
        vibration_busy = True

    if source == "ULTRA":
        last_ultra_vib_time = now
    else:
        last_camera_vib_time = now

    threading.Thread(
        target=_vibration_worker,
        args=(pin, pattern),
        daemon=True
    ).start()

    return True


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


def update_ultrasonic_if_needed(frame_count):
    global last_raw_left, last_raw_right
    global last_left_dist, last_right_dist
    global last_ultra_detected, last_ultra_direction, last_ultra_distance

    if frame_count % ULTRA_INTERVAL == 0:
        raw_left = get_distance_safe(TRIG_LEFT, ECHO_LEFT)
        time.sleep(0.005)
        raw_right = get_distance_safe(TRIG_RIGHT, ECHO_RIGHT)

        left_dist, right_dist = stabilize_ultrasonic(raw_left, raw_right)
        ultra_detected, ultra_direction, ultra_distance = judge_ultrasonic(
            left_dist,
            right_dist
        )

        last_raw_left = raw_left
        last_raw_right = raw_right
        last_left_dist = left_dist
        last_right_dist = right_dist
        last_ultra_detected = ultra_detected
        last_ultra_direction = ultra_direction
        last_ultra_distance = ultra_distance

    return (
        last_raw_left,
        last_raw_right,
        last_left_dist,
        last_right_dist,
        last_ultra_detected,
        last_ultra_direction,
        last_ultra_distance
    )


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
    if depth_map is None:
        return None

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


# =====================================================
# TTS 함수
# =====================================================
def make_message_candidate(source, direction, label=None, distance=None):
    direction_kr = {
        "LEFT": "좌측",
        "CENTER": "중앙",
        "RIGHT": "우측"
    }.get(direction, "중앙")

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

    if distance is not None and distance > 0:
        meter = distance / 100.0
        return f"{direction_kr} {meter:.1f}미터 앞 {label_kr}이 있습니다."

    return f"{direction_kr} 앞 {label_kr}이 있습니다."


def speak_tts(message):
    global last_tts_time, last_tts_message

    now = time.time()

    if now - last_tts_time < TTS_COOLDOWN:
        return

    if message == last_tts_message and now - last_tts_time < TTS_REPEAT_COOLDOWN:
        return

    def _speak():
        try:
            tts = gTTS(text=message, lang="ko")
            tts.save(TTS_PATH)

            subprocess.run(
                ["mpg123", "-q", TTS_PATH],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=6
            )
        except subprocess.TimeoutExpired:
            print("[TTS 오류] mpg123 재생 timeout")
        except Exception as e:
            print(f"[TTS 오류] {e}")

    threading.Thread(target=_speak, daemon=True).start()

    last_tts_time = now
    last_tts_message = message


def put_latest(q, item):
    """큐가 가득 차면 오래된 항목을 버리고 최신 항목만 유지한다."""
    try:
        if q.full():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                pass
        q.put_nowait(item)
    except queue.Full:
        pass


def generate_scene_caption(frame):
    if scene_caption_processor is None or scene_caption_model is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = scene_caption_processor(image, return_tensors="pt").to(scene_caption_device)

    with torch.no_grad():
        output = scene_caption_model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=3
        )

    caption = scene_caption_processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def clean_blip_caption(caption):
    if not caption:
        return ""

    text = caption.strip()
    text = text.replace(" ' s", "'s")
    text = " ".join(text.split())
    return text


def clean_qwen_output(text):
    if not text:
        return ""

    text = text.strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    # 모델이 불필요한 접두어를 붙이는 경우 제거
    prefixes = [
        "한국어 화면 해설:",
        "한국어 안내문:",
        "화면 해설:",
        "안내문:",
        "답변:",
        "출력:",
        "번역:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # 따옴표로 감싼 답변 정리
    text = text.strip('"\'“”‘’ ')

    # 너무 길게 말하면 보행 중 방해되므로 짧게 제한
    if len(text) > 80:
        text = text[:80].rstrip() + "..."

    return text


def contains_hangul(text):
    return bool(re.search(r"[가-힣]", text or ""))


def contains_cjk_han(text):
    """중국어/한자 계열 문자가 섞이면 True. 화면 해설 TTS에서는 사용하지 않는다."""
    return bool(re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", text or ""))


def is_valid_korean_scene_message(text):
    if not text:
        return False
    if contains_cjk_han(text):
        return False
    if not contains_hangul(text):
        return False
    # 영어 단어가 많이 섞이면 실패로 본다. 숫자/기호 정도는 허용.
    english_letters = len(re.findall(r"[A-Za-z]", text))
    if english_letters > 3:
        return False
    return True


def call_qwen_scene_api(messages):
    payload = {
        "model": "qwen",
        "messages": messages,
        "max_tokens": QWEN_MAX_TOKENS,
        "temperature": QWEN_TEMPERATURE,
        "stop": ["\n"],
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        QWEN_API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=QWEN_API_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")

    result = json.loads(raw)
    return clean_qwen_output(result["choices"][0]["message"]["content"])


def caption_to_scene_message(caption):
    """BLIP 영어 캡션을 localhost llama-server의 Qwen으로 한국어 화면 해설로 변환한다."""
    caption = clean_blip_caption(caption)

    if not caption:
        return "현재 화면을 설명하기 어렵습니다.", "ko"

    if not qwen_available:
        return caption, "en"

    system_prompt = (
        "너는 영어 이미지 캡션을 한국어 화면 해설로 바꾸는 번역기다. "
        "반드시 한국어로만 답하라. 중국어, 영어, 한자는 절대 사용하지 마라. "
        "직역하지 말고 자연스럽게 말하라. "
        "원문에 없는 물체, 위험, 거리, 방향은 추가하지 마라. "
        "출력은 한국어 한 문장만 하라."
    )

    # 1차: 짧은 프롬프트. curl 테스트에서 성공한 형태와 동일하게 유지.
    attempts = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"영어 캡션: {caption}\n한국어 화면 해설:"},
        ],
        # 2차: 중국어/한자 출력 방지용 few-shot. 1차 결과가 중국어로 튀면 재시도한다.
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "영어 캡션: a person standing in a room\n한국어 화면 해설:"},
            {"role": "assistant", "content": "방 안에 사람이 서 있습니다."},
            {"role": "user", "content": "영어 캡션: a cup on a table\n한국어 화면 해설:"},
            {"role": "assistant", "content": "테이블 위에 컵이 보입니다."},
            {"role": "user", "content": f"영어 캡션: {caption}\n한국어 화면 해설:"},
        ],
    ]

    try:
        last_message = ""

        for idx, messages in enumerate(attempts, start=1):
            message = call_qwen_scene_api(messages)
            last_message = message

            if QWEN_PRINT_DEBUG:
                print(f"[QWEN API SCENE TRY {idx}] {message}")

            if is_valid_korean_scene_message(message):
                return message, "ko"

            print(f"[QWEN API SKIP] 한국어 검증 실패. 재시도 또는 fallback: {message}")

        print(f"[QWEN API FALLBACK] 한국어 변환 실패. 영어 캡션 사용: {caption} / last={last_message}")
        return caption, "en"

    except urllib.error.URLError as e:
        print(f"[QWEN API 오류] llama-server 연결 실패. 영어 캡션 그대로 사용: {e}")
        return caption, "en"
    except Exception as e:
        print(f"[QWEN API 오류] 화면 해설 변환 실패. 영어 캡션 그대로 사용: {e}")
        return caption, "en"


def scene_speak_tts(message, lang="ko"):
    """일반 화면 해설 TTS. 위험 안내 직후에는 출력하지 않는다."""
    if not SCENE_CAPTION_TTS_ENABLE:
        print("[SCENE TTS SKIP] SCENE_CAPTION_TTS_ENABLE=False")
        return

    now = time.time()

    # 기존 위험/객체 TTS가 최근에 나간 경우 화면 해설은 양보한다.
    # 단, 테스트 편의를 위해 생략 사유를 항상 출력한다.
    remain = SCENE_TTS_SUPPRESS_AFTER_ALERT - (now - last_tts_time)
    if remain > 0:
        print(f"[SCENE TTS SKIP] 최근 위험/객체 TTS 때문에 화면 해설 생략: {remain:.1f}s 남음")
        return

    try:
        print(f"[SCENE TTS PLAY] {message}")
        tts = gTTS(text=message, lang=lang)
        tts.save(SCENE_TTS_PATH)
        subprocess.run(
            ["mpg123", "-q", SCENE_TTS_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8
        )
    except subprocess.TimeoutExpired:
        print("[SCENE TTS 오류] mpg123 재생 timeout")
    except Exception as e:
        print(f"[SCENE TTS 오류] {e}")


def make_scene_event_payload(caption, message, lang="ko"):
    return {
        "name": "화면 해설",
        "direction": "중앙",
        "distance": "-",
        "confidence": "-",
        "level": "INFO",
        "message": message,
        "source": "SCENE",
        "vibration": False,
        "pattern": "-",
        "caption": caption,
        "lang": lang,
        "timestamp": time.strftime("%H:%M:%S"),
    }


def scene_caption_worker(event_sender):
    global scene_caption_running

    while not scene_caption_stop:
        try:
            frame = scene_caption_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        scene_caption_running = True
        start = time.time()

        try:
            caption = generate_scene_caption(frame)
            elapsed = time.time() - start

            if caption:
                message, message_lang = caption_to_scene_message(caption)
                print(f"[SCENE CAPTION] {caption}")
                print(f"[SCENE MESSAGE] {message}")
                print(f"[SCENE LANG] {message_lang}")
                print(f"[SCENE TIME] caption elapsed: {elapsed:.2f}s")

                if SCENE_CAPTION_EVENT_ENABLE and event_sender is not None:
                    event_sender.send_event(make_scene_event_payload(caption, message, message_lang))

                put_latest(scene_tts_queue, (message, message_lang))

        except Exception as e:
            print(f"[SCENE CAPTION 오류] {e}")

        finally:
            scene_caption_running = False
            scene_caption_queue.task_done()


def scene_tts_worker():
    global scene_tts_running

    while not scene_caption_stop:
        try:
            item = scene_tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if isinstance(item, tuple):
            message, message_lang = item
        else:
            message, message_lang = item, "ko"

        scene_tts_running = True

        try:
            scene_speak_tts(message, message_lang)
        except Exception as e:
            print(f"[SCENE TTS WORKER 오류] {e}")
        finally:
            scene_tts_running = False
            scene_tts_queue.task_done()


def start_scene_caption_workers(event_sender):
    if not SCENE_CAPTION_ENABLE:
        return

    if scene_caption_processor is None or scene_caption_model is None:
        print("[SCENE] 화면 해설 모델이 없어 worker를 시작하지 않습니다.")
        return

    threading.Thread(
        target=scene_caption_worker,
        args=(event_sender,),
        daemon=True
    ).start()

    threading.Thread(
        target=scene_tts_worker,
        daemon=True
    ).start()

    print(f"화면 해설 worker 시작: {SCENE_CAPTION_INTERVAL:.1f}초마다 최신 프레임 처리")


def submit_scene_caption_if_needed(frame, ultra_detected=False, risk_result=None):
    global last_scene_caption_submit_time

    if not SCENE_CAPTION_ENABLE:
        return

    now = time.time()

    # 초음파 위험이나 카메라 HIGH 위험이 있으면 일반 화면 해설은 생략한다.
    # 여기서는 last_scene_caption_submit_time을 갱신하지 않는다.
    # 그래야 위험이 사라진 직후 다음 주기까지 불필요하게 기다리지 않는다.
    if ultra_detected:
        if SCENE_PRINT_DEBUG:
            print("[SCENE SKIP] 초음파 위험 감지 중이라 화면 해설 생략")
        return

    if vibration_busy:
        if SCENE_PRINT_DEBUG:
            print("[SCENE SKIP] 진동 출력 중이라 화면 해설 생략")
        return

    if risk_result is not None and risk_result.get("risk") == "HIGH":
        if SCENE_PRINT_DEBUG:
            print("[SCENE SKIP] 카메라 HIGH 위험 감지 중이라 화면 해설 생략")
        return

    if now - last_scene_caption_submit_time < SCENE_CAPTION_INTERVAL:
        return

    if scene_caption_running:
        if SCENE_PRINT_DEBUG:
            print("[SCENE SKIP] 이전 화면 해설 처리 중이라 이번 프레임 생략")
        return

    put_latest(scene_caption_queue, frame.copy())
    last_scene_caption_submit_time = now

    if SCENE_PRINT_DEBUG:
        print("[SCENE QUEUE] 화면 해설 프레임 제출")


# =====================================================
# 시각화
# =====================================================
def draw_result(frame, risk_result, yolo_boxes, ultra_info=None, confirm_count=0):
    h, w, _ = frame.shape

    if DRAW_GUIDE_LINES:
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
# FPS 계산 함수
# =====================================================
def update_average_fps(prev_time, fps_smooth):
    now = time.time()
    instant_fps = 1.0 / max(now - prev_time, 1e-6)

    if fps_smooth is None:
        fps_smooth = instant_fps
    else:
        fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1

    return now, fps_smooth


# =====================================================
# 메인
# =====================================================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

# 해상도 지정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# FFmpeg RTSP 송출 시작
stream_process = None

# WebSocket 이벤트 로그 전송 시작
event_sender = EventSender(EVENT_WS_URL)

if EVENT_LOG_ENABLE:
    event_sender.start()
    print(f"이벤트 로그 WebSocket 전송 시작: {EVENT_WS_URL}")

# 화면 해설 보조 worker 시작
start_scene_caption_workers(event_sender)

if STREAM_ENABLE:
    stream_process = start_ffmpeg_stream(
        STREAM_WIDTH,
        STREAM_HEIGHT,
        STREAM_FPS,
        STREAM_URL
    )
    print(f"MediaMTX RTSP 송출 시작: {STREAM_URL}")
    print(f"송출 설정: {STREAM_WIDTH}x{STREAM_HEIGHT} @ {STREAM_FPS}fps")

frame_count = 0
depth_map = None
prev_time = time.time()
fps_smooth = None

print("Vest Final Logic V6.6 Qwen API Scene Caption + HD 시작")
print("정책:")
print("- MiDaS는 3프레임마다 실행")
print("- 초음파는 3프레임마다 측정하고 직전 값을 재사용")
print("- 진동은 별도 스레드에서 실행하여 FPS 저하를 줄임")
print("- Depth 창 표시 기본 OFF")
print("- DEBUG_PRINT 기본 OFF")
print("- FFmpeg libx264로 MediaMTX RTSP 송출")
print(f"- 카메라 입력 해상도: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print(f"- RTSP 송출 해상도: {STREAM_WIDTH}x{STREAM_HEIGHT} @ {STREAM_FPS}fps")
print(f"- 개발용 좌/중/우 분할선 표시: {DRAW_GUIDE_LINES}")
print("- 종료: q")

try:
    while not shutdown_requested:
        ret, frame = cap.read()

        if not ret:
            print("프레임 읽기 실패")
            break

        h, w, _ = frame.shape

        # -----------------------------
        # 1. 초음파 측정 / 직전값 재사용
        # -----------------------------
        ultra_info = update_ultrasonic_if_needed(frame_count)

        (
            raw_left,
            raw_right,
            left_dist,
            right_dist,
            ultra_detected,
            ultra_direction,
            ultra_distance
        ) = ultra_info

        # -----------------------------
        # 2. 초음파 긴급 우선
        # -----------------------------
        if ultra_detected:
            pattern = "STRONG" if ultra_distance <= ULTRA_EMERGENCY_DIST else "SHORT"

            msg = make_message_candidate(
                "ULTRA",
                ultra_direction,
                label="unknown obstacle",
                distance=ultra_distance
            )

            debug_log(
                f"[초음파 긴급] L:{left_dist}cm R:{right_dist}cm | "
                f"방향:{ultra_direction} | 거리:{ultra_distance}cm | "
                f"패턴:{pattern} | 메시지:{msg}"
            )

            vibration_triggered = trigger_vibration(
                ultra_direction,
                pattern=pattern,
                source="ULTRA"
            )

            if vibration_triggered:
                level = "HIGH" if ultra_distance <= ULTRA_EMERGENCY_DIST else "MEDIUM"
                send_vibration_log_event(
                    event_sender=event_sender,
                    source="ULTRA",
                    direction=ultra_direction,
                    name="초음파 장애물",
                    distance=ultra_distance,
                    confidence="-",
                    level=level,
                    pattern=pattern
                )

            speak_tts(msg)

        # -----------------------------
        # 3. MiDaS 갱신
        # -----------------------------
        if frame_count % MIDAS_INTERVAL == 0 or depth_map is None:
            depth_map = estimate_depth(frame)

        # -----------------------------
        # 4. YOLO + MiDaS 매칭
        # -----------------------------
        depth_obstacle = find_near_obstacle_from_depth(depth_map, w, h)
        yolo_boxes = run_yolo(frame)
        risk_result = match_yolo_with_depth_obstacle(yolo_boxes, depth_obstacle)

        confirm_count = update_camera_confirm(risk_result)
        do_camera_vib, camera_pattern = should_camera_vibrate(risk_result, confirm_count)

        if risk_result is not None:
            camera_distance = None

            if ultra_direction == risk_result["direction"] and ultra_distance is not None:
                camera_distance = ultra_distance

            msg = make_message_candidate(
                "CAMERA",
                risk_result["direction"],
                label=risk_result["label"],
                distance=camera_distance
            )

            debug_log(
                f"[카메라 판단] 객체:{risk_result['label']} | "
                f"방향:{risk_result['direction']} | "
                f"matched:{risk_result['matched']} | "
                f"depth:{risk_result['depth_score']:.2f} | "
                f"near_ratio:{risk_result['near_ratio']:.2f} | "
                f"연속:{confirm_count} | "
                f"진동:{do_camera_vib} | "
                f"메시지후보:{msg}"
            )

            # 위험 객체: 진동 + TTS
            if not ultra_detected and do_camera_vib:
                vibration_triggered = trigger_vibration(
                    risk_result["direction"],
                    pattern=camera_pattern,
                    source="CAMERA"
                )

                if vibration_triggered:
                    camera_level = risk_result.get("risk", "MEDIUM")
                    if camera_level == "UNKNOWN":
                        camera_level = "MEDIUM"

                    send_vibration_log_event(
                        event_sender=event_sender,
                        source="CAMERA",
                        direction=risk_result["direction"],
                        name=label_to_korean(risk_result["label"]),
                        distance=camera_distance,
                        confidence=risk_result.get("conf"),
                        level=camera_level,
                        pattern=camera_pattern
                    )

                speak_tts(msg)

            # 사람: 진동 없이 TTS만 출력
            elif (
                not ultra_detected
                and risk_result["matched"]
                and risk_result["label"] == "person"
            ):
                speak_tts(msg)

        elif not ultra_detected:
            all_vibrations_off()

        # -----------------------------
        # 5. 5초 주기 이벤트 로그 전송
        # -----------------------------
        send_periodic_status_event(
            event_sender=event_sender,
            ultra_detected=ultra_detected,
            ultra_direction=ultra_direction,
            ultra_distance=ultra_distance,
            risk_result=risk_result
        )

        # -----------------------------
        # 6. 화면 해설 캡셔닝 요청
        # -----------------------------
        submit_scene_caption_if_needed(
            frame,
            ultra_detected=ultra_detected,
            risk_result=risk_result
        )

        # -----------------------------
        # 7. FPS 표시
        # -----------------------------
        prev_time, fps_smooth = update_average_fps(prev_time, fps_smooth)

        cv2.putText(
            frame,
            f"FPS:{fps_smooth:.2f} | DEVICE:{device}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # -----------------------------
        # 8. 화면 표시용 결과 그리기
        # -----------------------------
        frame = draw_result(
            frame,
            risk_result,
            yolo_boxes,
            ultra_info=ultra_info,
            confirm_count=confirm_count
        )

        # -----------------------------
        # 9. MediaMTX RTSP 송출
        # -----------------------------
        if STREAM_ENABLE and stream_process is not None:
            try:
                stream_frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
                stream_process.stdin.write(stream_frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg 스트림 연결이 끊겼습니다.")
                stream_process = None
            except Exception as e:
                print(f"FFmpeg 송출 오류: {e}")
                stream_process = None

        # -----------------------------
        # 10. 로컬 화면 표시
        # -----------------------------
        cv2.imshow("Vest Final Logic V6.5 Qwen API Scene Caption", frame)

        if SHOW_DEPTH_WINDOW and depth_map is not None:
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imshow("MiDaS Depth", depth_vis)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            request_shutdown()
            break

except KeyboardInterrupt:
    print("\n종료합니다.")

finally:
    scene_caption_stop = True
    shutdown_requested = True

    # TTS 재생 프로세스가 종료를 붙잡는 경우를 방지
    try:
        subprocess.run(["pkill", "-f", "mpg123 -q /tmp/vest"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()

    if stream_process is not None:
        try:
            stream_process.stdin.close()
            stream_process.terminate()
            try:
                stream_process.wait(timeout=2)
            except Exception:
                stream_process.kill()
        except Exception:
            pass

    if EVENT_LOG_ENABLE:
        event_sender.stop()

    all_vibrations_off()
    GPIO.cleanup()
