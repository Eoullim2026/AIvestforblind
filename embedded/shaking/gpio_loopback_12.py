import Jetson.GPIO as GPIO
import time

TRIG = 12
ECHO = 16

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

try:
    while True:
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.1)
        print("PIN 12 HIGH, ECHO =", GPIO.input(ECHO))

        GPIO.output(TRIG, GPIO.LOW)
        time.sleep(0.1)
        print("PIN 12 LOW,  ECHO =", GPIO.input(ECHO))

except KeyboardInterrupt:
    print("종료")

finally:
    GPIO.cleanup()
