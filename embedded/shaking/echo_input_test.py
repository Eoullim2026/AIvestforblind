import Jetson.GPIO as GPIO
import time

ECHO = 16  # Jetson 물리 16번 핀

GPIO.setmode(GPIO.BOARD)
GPIO.setup(ECHO, GPIO.IN)

try:
    while True:
        value = GPIO.input(ECHO)
        print(value)
        time.sleep(0.2)

except KeyboardInterrupt:
    print("종료")

finally:
    GPIO.cleanup()
