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
        GPIO.output(TRIG, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG, GPIO.LOW)

        start = time.time()
        detected = False

        while time.time() - start < 0.1:
            val = GPIO.input(ECHO)
            if val == 1:
                detected = True
                print("ECHO HIGH detected")
                break

        if not detected:
            print("ECHO never went HIGH")

        time.sleep(0.5)

except KeyboardInterrupt:
    pass

finally:
    GPIO.cleanup()
