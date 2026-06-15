import Jetson.GPIO as GPIO
import time

# GPIO.BOARD 기준: Jetson 물리 핀 번호
TRIG1 = 11
ECHO1 = 16

TRIG2 = 12
ECHO2 = 18

TIMEOUT = 0.025      # 약 4m 정도까지 측정
INTERVAL = 0.12      # 센서 간 간섭 방지 대기시간


def measure_distance(trig, echo):
    # 이전 ECHO 신호가 남아 있으면 LOW가 될 때까지 기다림
    wait_start = time.time()
    while GPIO.input(echo) == 1:
        if time.time() - wait_start > TIMEOUT:
            return None

    # 초음파 발사 전 안정화
    GPIO.output(trig, GPIO.LOW)
    time.sleep(0.000005)

    # 10us 트리거 신호
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)

    # ECHO가 HIGH가 될 때까지 대기
    start_time = time.time()
    pulse_start = None

    while GPIO.input(echo) == 0:
        pulse_start = time.time()

        if pulse_start - start_time > TIMEOUT:
            return None

    # ECHO가 LOW가 될 때까지 대기
    start_time = time.time()
    pulse_end = None

    while GPIO.input(echo) == 1:
        pulse_end = time.time()

        if pulse_end - start_time > TIMEOUT:
            return None

    if pulse_start is None or pulse_end is None:
        return None

    # 거리 계산
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2

    # HC-SR04 유효 범위 필터링
    if distance < 2 or distance > 400:
        return None

    return distance


GPIO.setmode(GPIO.BOARD)

GPIO.setup(TRIG1, GPIO.OUT)
GPIO.setup(ECHO1, GPIO.IN)

GPIO.setup(TRIG2, GPIO.OUT)
GPIO.setup(ECHO2, GPIO.IN)

GPIO.output(TRIG1, GPIO.LOW)
GPIO.output(TRIG2, GPIO.LOW)

time.sleep(1)

try:
    while True:
        distance1 = measure_distance(TRIG1, ECHO1)
        time.sleep(INTERVAL)

        distance2 = measure_distance(TRIG2, ECHO2)
        time.sleep(INTERVAL)

        if distance1 is None:
            result1 = "센서1: 측정 실패"
        else:
            result1 = f"센서1: {distance1:.2f} cm"

        if distance2 is None:
            result2 = "센서2: 측정 실패"
        else:
            result2 = f"센서2: {distance2:.2f} cm"

        print(f"{result1} | {result2}")

        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n종료")

finally:
    GPIO.cleanup()
