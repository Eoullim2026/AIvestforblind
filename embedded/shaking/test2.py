import Jetson.GPIO as GPIO
import time

TRIG = 11
ECHO = 16

GPIO.setmode(GPIO.BOARD)

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, GPIO.LOW)
time.sleep(1)

try:
    while True:
        # 초음파 발사
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)  # 10us
        GPIO.output(TRIG, GPIO.LOW)

        # ECHO가 HIGH가 될 때까지 대기
        timeout = time.time()
        pulse_start = None

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()

            if pulse_start - timeout > 0.025:
                print("Timeout waiting for HIGH")
                break

        if pulse_start is None or GPIO.input(ECHO) == 0:
            time.sleep(0.3)
            continue

        # ECHO가 LOW가 될 때까지 대기
        timeout = time.time()
        pulse_end = None

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

            if pulse_end - timeout > 0.025:
                print("Timeout waiting for LOW")
                break

        if pulse_end is None or GPIO.input(ECHO) == 1:
            print("측정 실패")
            time.sleep(0.3)
            continue

        # 거리 계산
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 34300 / 2

        # HC-SR04 유효 거리 범위 필터링
        if distance < 2 or distance > 400:
            print("측정 실패")
            time.sleep(0.3)
            continue

        print(f"거리: {distance:.2f} cm")

        time.sleep(0.3)

except KeyboardInterrupt:
    print("종료")

finally:
    GPIO.cleanup()
