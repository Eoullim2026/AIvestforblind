import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 초음파 센서 핀
TRIG1, ECHO1 = 11, 16  # 왼쪽 센서
TRIG2, ECHO2 = 12, 18  # 오른쪽 센서

# 진동모듈 핀
VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([TRIG1, TRIG2], GPIO.OUT)
GPIO.setup([ECHO1, ECHO2], GPIO.IN)
GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

# 거리 기준
DETECT_MAX_DIST = 100.0   # 100cm 이하에서 진동
SIDE_TOLERANCE = 25.0     # 좌우 차이가 25cm 이하면 중앙 장애물


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


def all_vibrations_off():
    GPIO.output([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.LOW)


def trigger_vibration(vib_pin):
    GPIO.output(vib_pin, GPIO.HIGH)


def handle_detection(left_dist, right_dist):
    left_valid = left_dist > 0
    right_valid = right_dist > 0

    left_detected = left_valid and left_dist <= DETECT_MAX_DIST
    right_detected = right_valid and right_dist <= DETECT_MAX_DIST

    if not left_valid and not right_valid:
        print("⚠️ [측정 실패] 양쪽 센서 응답 없음")
        return

    if left_detected and right_detected:
        if abs(left_dist - right_dist) <= SIDE_TOLERANCE:
            print("🚨 [중앙 장애물 감지] → 중앙 진동")
            trigger_vibration(VIB_CENTER)
        elif left_dist < right_dist:
            print("👈 [왼쪽 장애물 감지] → 왼쪽 진동")
            trigger_vibration(VIB_LEFT)
        else:
            print("👉 [오른쪽 장애물 감지] → 오른쪽 진동")
            trigger_vibration(VIB_RIGHT)

    elif left_detected:
        print("👈 [왼쪽 장애물 감지] → 왼쪽 진동")
        trigger_vibration(VIB_LEFT)

    elif right_detected:
        print("👉 [오른쪽 장애물 감지] → 오른쪽 진동")
        trigger_vibration(VIB_RIGHT)

    else:
        print("🟢 전방 안전")


print("🚀 초음파 기반 진동 피드백 시스템 시작")
time.sleep(1)

try:
    while True:
        left_dist = get_distance_safe(TRIG1, ECHO1)
        time.sleep(0.03)

        right_dist = get_distance_safe(TRIG2, ECHO2)
        time.sleep(0.03)

        print(f"👀 왼쪽: {left_dist}cm | 오른쪽: {right_dist}cm")

        all_vibrations_off()
        handle_detection(left_dist, right_dist)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 시스템 종료")

finally:
    all_vibrations_off()
    GPIO.cleanup()
