import RPi.GPIO as GPIO
import time

# =========================
# GPIO PIN CONFIG - BOARD MODE
# =========================
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# Ultrasonic sensors
LEFT_TRIG = 11
LEFT_ECHO = 16

RIGHT_TRIG = 12
RIGHT_ECHO = 18

# Vibration motors
LEFT_VIB = 21
CENTER_VIB = 23
RIGHT_VIB = 24

# Distance thresholds in cm
NEAR_DISTANCE = 100      # 강한 위험
MID_DISTANCE = 200       # 중간 위험
MAX_DISTANCE = 300       # 이 이상은 무시

# Vibration timing
STRONG_ON = 0.35
STRONG_OFF = 0.15

MID_ON = 0.2
MID_OFF = 0.3

WEAK_ON = 0.12
WEAK_OFF = 0.5


# =========================
# GPIO SETUP
# =========================
def setup_gpio():
    # Ultrasonic setup
    GPIO.setup(LEFT_TRIG, GPIO.OUT)
    GPIO.setup(LEFT_ECHO, GPIO.IN)

    GPIO.setup(RIGHT_TRIG, GPIO.OUT)
    GPIO.setup(RIGHT_ECHO, GPIO.IN)

    GPIO.output(LEFT_TRIG, GPIO.LOW)
    GPIO.output(RIGHT_TRIG, GPIO.LOW)

    # Vibration setup
    GPIO.setup(LEFT_VIB, GPIO.OUT)
    GPIO.setup(CENTER_VIB, GPIO.OUT)
    GPIO.setup(RIGHT_VIB, GPIO.OUT)

    all_vibration_off()

    time.sleep(0.5)


# =========================
# VIBRATION CONTROL
# =========================
def all_vibration_off():
    GPIO.output(LEFT_VIB, GPIO.LOW)
    GPIO.output(CENTER_VIB, GPIO.LOW)
    GPIO.output(RIGHT_VIB, GPIO.LOW)


def vibrate(pin, on_time, off_time):
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(on_time)
    GPIO.output(pin, GPIO.LOW)
    time.sleep(off_time)


def vibrate_left(level):
    if level == "HIGH":
        vibrate(LEFT_VIB, STRONG_ON, STRONG_OFF)
    elif level == "MID":
        vibrate(LEFT_VIB, MID_ON, MID_OFF)
    elif level == "LOW":
        vibrate(LEFT_VIB, WEAK_ON, WEAK_OFF)


def vibrate_right(level):
    if level == "HIGH":
        vibrate(RIGHT_VIB, STRONG_ON, STRONG_OFF)
    elif level == "MID":
        vibrate(RIGHT_VIB, MID_ON, MID_OFF)
    elif level == "LOW":
        vibrate(RIGHT_VIB, WEAK_ON, WEAK_OFF)


def vibrate_center(level):
    if level == "HIGH":
        vibrate(CENTER_VIB, STRONG_ON, STRONG_OFF)
    elif level == "MID":
        vibrate(CENTER_VIB, MID_ON, MID_OFF)
    elif level == "LOW":
        vibrate(CENTER_VIB, WEAK_ON, WEAK_OFF)


# =========================
# ULTRASONIC MEASUREMENT
# =========================
def measure_distance(trig_pin, echo_pin, timeout=0.03):
    """
    HC-SR04 distance measurement.
    Returns distance in cm, or None on failure.
    """

    # Trigger pulse
    GPIO.output(trig_pin, GPIO.LOW)
    time.sleep(0.0002)

    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.LOW)

    start_time = time.time()

    # Wait for ECHO HIGH
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
        if pulse_start - start_time > timeout:
            return None

    # Wait for ECHO LOW
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()
        if pulse_end - pulse_start > timeout:
            return None

    pulse_duration = pulse_end - pulse_start

    # Speed of sound: 34300 cm/s
    distance = pulse_duration * 34300 / 2

    if distance <= 0 or distance > MAX_DISTANCE:
        return None

    return round(distance, 2)


# =========================
# RISK LEVEL
# =========================
def get_level(distance):
    if distance is None:
        return "NONE"

    if distance <= NEAR_DISTANCE:
        return "HIGH"
    elif distance <= MID_DISTANCE:
        return "MID"
    elif distance <= MAX_DISTANCE:
        return "LOW"
    else:
        return "NONE"


# =========================
# MAIN LOGIC
# =========================
def main():
    setup_gpio()

    print("====================================")
    print("Ultrasonic + Vibration Test Started")
    print("GPIO Mode: BOARD")
    print("------------------------------------")
    print(f"LEFT  SENSOR: TRIG={LEFT_TRIG}, ECHO={LEFT_ECHO}")
    print(f"RIGHT SENSOR: TRIG={RIGHT_TRIG}, ECHO={RIGHT_ECHO}")
    print(f"LEFT VIB={LEFT_VIB}, CENTER VIB={CENTER_VIB}, RIGHT VIB={RIGHT_VIB}")
    print("Ctrl+C to stop")
    print("====================================")

    try:
        while True:
            left_dist = measure_distance(LEFT_TRIG, LEFT_ECHO)
            time.sleep(0.06)

            right_dist = measure_distance(RIGHT_TRIG, RIGHT_ECHO)
            time.sleep(0.06)

            left_level = get_level(left_dist)
            right_level = get_level(right_dist)

            left_text = f"{left_dist} cm" if left_dist is not None else "측정 실패"
            right_text = f"{right_dist} cm" if right_dist is not None else "측정 실패"

            print(
                f"좌측 센서: {left_text:>10} | 좌측 위험도: {left_level:>4} || "
                f"우측 센서: {right_text:>10} | 우측 위험도: {right_level:>4}"
            )

            # Both sides dangerous
            if left_level != "NONE" and right_level != "NONE":
                if left_dist is not None and right_dist is not None:
                    diff = abs(left_dist - right_dist)

                    # 둘 다 비슷하게 가까우면 중앙 진동
                    if diff <= 25:
                        level = "HIGH" if min(left_dist, right_dist) <= NEAR_DISTANCE else "MID"
                        print(f"[VIB] CENTER / level={level}")
                        vibrate_center(level)

                    # 왼쪽이 더 가까움
                    elif left_dist < right_dist:
                        print(f"[VIB] LEFT / level={left_level}")
                        vibrate_left(left_level)

                    # 오른쪽이 더 가까움
                    else:
                        print(f"[VIB] RIGHT / level={right_level}")
                        vibrate_right(right_level)

            # Left only
            elif left_level != "NONE":
                print(f"[VIB] LEFT / level={left_level}")
                vibrate_left(left_level)

            # Right only
            elif right_level != "NONE":
                print(f"[VIB] RIGHT / level={right_level}")
                vibrate_right(right_level)

            else:
                all_vibration_off()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nTest stopped by user.")

    finally:
        all_vibration_off()
        GPIO.cleanup()
        print("GPIO cleanup complete.")


if __name__ == "__main__":
    main()
