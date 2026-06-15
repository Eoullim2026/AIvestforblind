import Jetson.GPIO as GPIO
import time

TRIG = 11
ECHO = 16

GPIO.setmode(GPIO.BOARD)

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, GPIO.LOW)
time.sleep(2)

try:
    while True:
        # 초음파 발사
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        # ECHO가 HIGH가 될 때까지 대기
        timeout = time.time()
        pulse_start = None

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()

            if pulse_start - timeout > 0.02:
                print("Timeout waiting for HIGH")
                break

        # HIGH를 못 받았으면 이번 측정 건너뛰기
        if pulse_start is None or GPIO.input(ECHO) == 0:
            time.sleep(1)
            continue

        # ECHO가 LOW가 될 때까지 대기
        timeout = time.time()
        pulse_end = None

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

            if pulse_end - timeout > 0.02:
                print("Timeout waiting for LOW")
                break

        # LOW를 못 받았으면 이번 측정 건너뛰기
        if pulse_end is None or GPIO.input(ECHO) == 1:
            time.sleep(1)
            continue

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 34300 / 2

        print(f"거리: {distance:.2f} cm")

        time.sleep(1)

except KeyboardInterrupt:
    print("종료")

finally:
    GPIO.cleanup()
