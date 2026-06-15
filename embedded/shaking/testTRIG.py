import Jetson.GPIO as GPIO
import time

TRIG = 11

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT)

try:
    while True:
        GPIO.output(TRIG, GPIO.HIGH)
        print("HIGH")
        time.sleep(1)

        GPIO.output(TRIG, GPIO.LOW)
        print("LOW")
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
