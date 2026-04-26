"""
PhytoClone — ESP32 WROOM Receiver (Plant B)
============================================
Matched to sender_differential.py — receives raw and baseline
values sent by the ESP32-S3 differential sender.

No transistor. Low duty cycle PWM direct to RC filter and needle.

Hardware:
  PWM Pin 18 -> [1.5kΩ] -> junction -> injection needle (+)
  junction   -> [10µF]  -> GND
  injection needle (-)  -> GND

Duty cycle kept at 0-30 out of 1023 (~0-97mV at RC junction).
Plant B injection signal tracks Plant A deviation in real time.

Steps:
  1. Run this first — it prints the WROOM MAC address
  2. Copy that MAC into sender_differential.py RECEIVER_MAC
  3. Flash sender onto ESP32-S3
"""

import network
import espnow
from machine import Pin, PWM
import time

PWM_PIN      = 18
PWM_FREQ     = 1000

MAX_DUTY     = 12   # 30/1023 = ~97mV max at RC junction
MIN_DUTY     = 0
CENTRE_DUTY  = 6    # resting duty when Plant A is unstressed (~48mV)

BASELINE_LEN = 300

# How sensitive duty response is to Plant A deviation.
# Lower = more sensitive. Raise if duty swings too wildly.
# Lower if signals are weak and you want more injection response.
SCALE_COUNTS = 400


def init_espnow():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    mac  = wlan.config("mac")

    print("=" * 52)
    print("WROOM MAC — paste into sender_differential.py:")
    print("  Readable:    ", ":".join("{:02x}".format(b) for b in mac))
    print('  Bytes:       ', "b\"" + "".join("\\x{:02x}".format(b) for b in mac) + "\"")
    print("=" * 52)
    print()

    en = espnow.ESPNow()
    en.active(True)
    return en


def deviation_to_duty(raw, base):
    """
    Map Plant A deviation to PWM duty 0-30.

    Positive deviation (depolarisation) -> higher duty -> more injection
    Negative deviation (hyperpolarisation) -> lower duty -> less injection
    No deviation -> CENTRE_DUTY -> stable resting injection voltage

    Tune SCALE_COUNTS to control sensitivity:
      2000 = moderate — 2000 count swing gives full duty range
      1000 = sensitive — 1000 count swing gives full duty range
      3000 = gentle    — 3000 count swing gives full duty range
    """
    deviation = raw - base
    swing     = CENTRE_DUTY
    duty      = CENTRE_DUTY + int((deviation / SCALE_COUNTS) * swing)
    return max(MIN_DUTY, min(MAX_DUTY, duty))


def rolling_baseline(buf, val):
    buf.pop(0)
    buf.append(val)
    return sum(buf) // len(buf)


def main():
    en  = init_espnow()
    pwm = PWM(Pin(PWM_PIN), freq=PWM_FREQ)
    pwm.duty(CENTRE_DUTY)

    baseline_b = [0] * BASELINE_LEN
    sample     = 0
    last_duty  = CENTRE_DUTY

    print("PhytoClone receiver — matched to differential sender")
    print("PWM Pin:", PWM_PIN, "| Duty range: 0 -", MAX_DUTY, "/ 1023")
    print("Max RC junction voltage: ~{}mV".format(round((MAX_DUTY / 1023) * 3300)))
    print()
    print("Waiting for ESP-NOW packets from ESP32-S3...")
    print()

    while True:
        try:
            host, msg = en.recv(timeout_ms=300)

            if msg is None:
                continue

            parts = msg.decode().strip().split(",")
            if len(parts) != 2:
                continue

            raw_a  = int(parts[0])
            base_a = int(parts[1])
            dev_a  = raw_a - base_a

            duty   = deviation_to_duty(raw_a, base_a)
            pwm.duty(duty)
            last_duty = duty

            sample += 1
            if sample % 8 == 0:
                mv_out   = round((duty / 1023) * 3300)
                changed  = "+" if duty > CENTRE_DUTY else ("-" if duty < CENTRE_DUTY else "=")
                print("#{:04d}  A dev:{:+6d}  duty:{:3d}/{}  [{}]  RC out: ~{}mV".format(
                    sample, dev_a, duty, MAX_DUTY, changed, mv_out))

        except Exception as e:
            print("Recv error:", e)
            time.sleep_ms(100)


main()