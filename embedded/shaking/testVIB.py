import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 무조건 뚫리는 청정 GPIO 핀으로 변경
VIB_LEFT = 7
VIB_CENTER = 29
VIB_RIGHT = 31

GPIO.setup([VIB_LEFT, VIB_CENTER, VIB_RIGHT], GPIO.OUT, initial=GPIO.LOW)

print("🚀 [안전 핀 전환] 7, 29, 31번 핀 기준 진동 테스트 시작...")

try:
    while True:
        print("👈 1번 모터 (7번 핀) ON")
        GPIO.output(VIB_LEFT, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(VIB_LEFT, GPIO.LOW)
        time.sleep(0.2)

        print("  2번 모터 (29번 핀) ON")
        GPIO.output(VIB_CENTER, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(VIB_CENTER, GPIO.LOW)
        time.sleep(0.2)

        print("👉 3번 모터 (31번 핀) ON")
        GPIO.output(VIB_RIGHT, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(VIB_RIGHT, GPIO.LOW)
        time.sleep(1.5)

except KeyboardInterrupt:
    GPIO.cleanup()
