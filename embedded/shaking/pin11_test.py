import Jetson.GPIO as GPIO
import time

PIN = 11

GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(PIN, GPIO.HIGH)
        print("PIN 11 HIGH")
        time.sleep(3)

        GPIO.output(PIN, GPIO.LOW)
        print("PIN 11 LOW")
        time.sleep(3)

except KeyboardInterrupt:
    print("종료")

finally:
    GPIO.cleanup()
