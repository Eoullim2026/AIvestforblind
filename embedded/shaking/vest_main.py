import Jetson.GPIO as GPIO
import time

# =========================
# GPIO 초기화
# =========================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# =========================
# 하드웨어 핀 할당
# =========================
TRIG1, ECHO1 = 11, 16  # 왼쪽 센서
TRIG2, ECHO2 = 12, 18  # 오른쪽 센서

VIB_LEFT = 21    # 왼쪽 진동
VIB_CENTER = 23  # 중앙 진동
VIB_RIGHT = 24   # 오른쪽 진동

# =========================
# GPIO 설정
# =========================
GPIO.setup([TRIG1, TRIG2], GPIO.OUT)
GPIO.setup([ECHO1, ECHO2], GPIO.IN)
GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

# =========================
# 거리 기준 설정
# =========================
DETECT_MAX_DIST = 200.0   # 2m 이내 감지
EMERGENCY_DIST = 100.0    # 1m 이내 연속 진동
SIDE_TOLERANCE = 25.0     # 좌우 차이가 25cm 이하면 중앙 장애물


def get_distance_safe(trig, echo):
    """무한 루프 방지 장치가 탑재된 안전한 거리 측정 함수"""

    GPIO.output(trig, False)
    time.sleep(0.000002)

    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    # Echo HIGH 대기
    t_start = time.time()
    pulse_start = time.time()

    while GPIO.input(echo) == 0:
        pulse_start = time.time()
        if pulse_start - t_start > 0.02:
            return -1

    # Echo LOW 대기
    t_start = time.time()
    pulse_end = time.time()

    while GPIO.input(echo) == 1:
        pulse_end = time.time()
        if pulse_end - t_start > 0.02:
            return -1

    duration = pulse_end - pulse_start
    distance = duration * 17150

    return round(distance, 1)


def all_vibrations_off():
    """모든 진동 모듈 끄기"""
    GPIO.output([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.LOW)


def trigger_vibration(vib_pin, distance):
    """거리별 진동 패턴 제어"""

    if distance <= EMERGENCY_DIST:
        # 1m 이내: 연속 진동
        GPIO.output(vib_pin, GPIO.HIGH)

    else:
        # 1m~2m: 짧게 끊어지는 진동
        GPIO.output(vib_pin, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(vib_pin, GPIO.LOW)


def handle_detection(left_dist, right_dist):
    """좌우 거리값을 기반으로 위험 방향 판단 및 진동 제어"""

    left_valid = left_dist > 0
    right_valid = right_dist > 0

    left_detected = left_valid and (left_dist <= DETECT_MAX_DIST)
    right_detected = right_valid and (right_dist <= DETECT_MAX_DIST)

    # 둘 다 측정 실패
    if not left_valid and not right_valid:
        print("⚠️ [측정 실패] 양쪽 센서 응답 없음")
        return

    # 둘 다 감지됨
    if left_detected and right_detected:
        if abs(left_dist - right_dist) <= SIDE_TOLERANCE:
            print("🚨 [중앙 장애물 발견]")
            trigger_vibration(VIB_CENTER, min(left_dist, right_dist))

        elif left_dist < right_dist:
            print("👈 [왼쪽 장애물 가까움]")
            trigger_vibration(VIB_LEFT, left_dist)

        else:
            print("👉 [오른쪽 장애물 가까움]")
            trigger_vibration(VIB_RIGHT, right_dist)

    # 왼쪽만 감지됨
    elif left_detected:
        print("👈 [왼쪽 장애물 발견]")
        trigger_vibration(VIB_LEFT, left_dist)

    # 오른쪽만 감지됨
    elif right_detected:
        print("👉 [오른쪽 장애물 발견]")
        trigger_vibration(VIB_RIGHT, right_dist)

    # 감지 없음
    else:
        print("🟢 전방 안전 구역")


print("🚀 [안전 모드] 초음파 기반 진동 피드백 시스템 시작...")
time.sleep(1)

try:
    while True:
        # 1. 좌우 거리 측정
        left_dist = get_distance_safe(TRIG1, ECHO1)
        time.sleep(0.03)

        right_dist = get_distance_safe(TRIG2, ECHO2)
        time.sleep(0.03)

        # 2. 로그 출력
        print(f"👀 [실시간 거리] 왼쪽: {left_dist}cm | 오른쪽: {right_dist}cm")

        # 3. 진동 초기화
        all_vibrations_off()

        # 4. 위험 판단 및 진동 제어
        handle_detection(left_dist, right_dist)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 시스템을 안전하게 종료합니다.")

finally:
    all_vibrations_off()
    GPIO.cleanup()
