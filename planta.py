# ============================================================
#  PhytoClone — ESP32-S3 Sender (Plant A)
#  With Plant B response monitor on A2-A3 differential
#
#  Hardware:
#    ADS1115 VDD  -> 3.3V
#    ADS1115 GND  -> GND
#    ADS1115 ADDR -> GND  (address 0x48)
#    ADS1115 SCL  -> Pin 9
#    ADS1115 SDA  -> Pin 8
#
#  Plant A electrodes (signal source):
#    Signal needle  -> [1.5kΩ] -> junction -> ADS1115 A0
#    junction       -> [10µF]  -> GND
#    Ref needle     -> ADS1115 A1
#    Soil ground    -> GND
#    Read as: A0 - A1 differential
#
#  Plant B monitor electrodes (response reader):
#    Monitor needle 1 -> ADS1115 A2   (petiole, 10-15cm above injection)
#    Monitor needle 2 -> ADS1115 A3   (stem, 5cm below that petiole)
#    Read as: A2 - A3 differential
#    Place monitor needles AS FAR as possible from injection needles
#    to avoid injection voltage coupling into the reading
#
#  Steps:
#    1. Flash receiver onto WROOM first — copy MAC it prints
#    2. Paste MAC into RECEIVER_MAC below
#    3. Flash this onto ESP32-S3
# ============================================================

import network
import espnow
from machine import I2C, Pin
import time

RECEIVER_MAC  = b'\x00K\x12;\x02X'   # paste WROOM MAC here

ADS_ADDR      = 0x48
BASELINE_LEN  = 60

# Gain index for all channels
# 4 = +-0.512V (15.625 uV/bit) — good starting point
# 5 = +-0.256V (7.8125 uV/bit) — use if signals are weak
GAIN_IDX      = 5

# Event detection thresholds (in raw counts)
# At gain 5: 1 count = 7.8125 uV
# 200 counts = ~1.56mV  (weak signal)
# 500 counts = ~3.9mV   (clear signal)
# 1000 counts = ~7.8mV  (strong signal)
STRESS_THRESH  = 500    # Plant A deviation to flag stress event
RESPONSE_THRESH = 300   # Plant B deviation to flag a biological response

# ============================================================
#  GAIN TABLE
# ============================================================
LSB_UV = {0: 187.5, 1: 125.0, 2: 62.5,
           3: 31.25, 4: 15.625, 5: 7.8125}

def counts_to_mv(counts):
    return counts * LSB_UV[GAIN_IDX] / 1000.0

# ============================================================
#  ESP-NOW INIT
# ============================================================
def init_espnow():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.disconnect()
    en = espnow.ESPNow()
    en.active(True)
    en.add_peer(RECEIVER_MAC)
    print("ESP-NOW initialised")
    print("Sending to:", ':'.join('{:02x}'.format(b) for b in RECEIVER_MAC))
    return en

# ============================================================
#  ADS1115 SINGLE-ENDED READ
#  Reads one channel against GND
#  Used to build differential pairs manually so we can read
#  A0, A1, A2, A3 independently then subtract
# ============================================================
def ads_read_single(i2c, channel):
    # MUX bits for single-ended channels
    # Sits in config byte 1 bits [14:12]
    mux = {0: 0x40, 1: 0x50, 2: 0x60, 3: 0x70}

    # Gain bits [11:9] split across both config bytes
    gain_bits = {0: 0x00, 1: 0x02, 2: 0x04,
                 3: 0x06, 4: 0x08, 5: 0x0A}

    # Build config register (16-bit, sent as 2 bytes)
    # Byte 1: OS=1 (start conv) | MUX[2:0] | PGA[2] | MODE=1 (single shot)
    # Byte 2: PGA[1:0]=0 | DR=100SPS | COMP disabled
    gb = gain_bits[GAIN_IDX]
    b1 = 0x80 | mux[channel] | ((gb >> 1) & 0x07)
    b2 = ((gb & 0x01) << 7) | 0x83

    i2c.writeto_mem(ADS_ADDR, 0x01, bytes([b1, b2]))

    # Wait for conversion — at 128SPS one conversion = ~8ms
    # Using 20ms for safety margin
    time.sleep_ms(20)

    raw = i2c.readfrom_mem(ADS_ADDR, 0x00, 2)
    val = (raw[0] << 8) | raw[1]
    return val - 65536 if val > 32767 else val

# ============================================================
#  DIFFERENTIAL READ
#  Reads two channels and returns the difference
#  A0-A1 for Plant A, A2-A3 for Plant B monitor
# ============================================================
def ads_read_diff(i2c, ch_pos, ch_neg):
    pos = ads_read_single(i2c, ch_pos)
    neg = ads_read_single(i2c, ch_neg)
    return pos - neg

# ============================================================
#  ROLLING BASELINE
# ============================================================
def rolling_baseline(buf, val):
    buf.pop(0)
    buf.append(val)
    return sum(buf) // len(buf)

# ============================================================
#  MAIN
# ============================================================
def main():
    i2c = I2C(0, scl=Pin(9), sda=Pin(8), freq=100000)
    found = i2c.scan()
    print("I2C scan:", [hex(d) for d in found])

    if ADS_ADDR not in found:
        print("ERROR: ADS1115 not found — check VDD, GND, ADDR, SCL, SDA")
        return

    en = init_espnow()

    # Separate baselines for Plant A and Plant B monitor
    baseline_a = [0] * BASELINE_LEN
    baseline_b = [0] * BASELINE_LEN

    sample = 0

    print()
    print("PhytoClone sender — Plant A source + Plant B monitor")
    print("Gain index:", GAIN_IDX, "| Resolution:", LSB_UV[GAIN_IDX], "uV/bit")
    print("Plant A: A0(+) - A1(-) differential")
    print("Plant B monitor: A2(+) - A3(-) differential")
    print()
    print("Monitor needle placement reminder:")
    print("  A2 -> petiole 10-15cm ABOVE Plant B injection site")
    print("  A3 -> stem 5cm below that petiole")
    print("  Keep monitor needles far from injection needles!")
    print()
    print("Settling baseline — do not touch either plant for",
          BASELINE_LEN, "samples (~{}s)...".format(
          int(BASELINE_LEN * 0.165)))
    print()
    print("{:<6} {:<12} {:<10} {:<12} {:<10} {}".format(
        "SAMPLE", "A_mV", "A_dev", "B_mon_mV", "B_dev", "STATUS"))
    print("-" * 70)

    while True:
        # Read Plant A differential (A0 - A1)
        raw_a  = ads_read_diff(i2c, 0, 1)

        # Read Plant B monitor differential (A2 - A3)
        raw_b  = ads_read_diff(i2c, 2, 3)

        # Update rolling baselines
        base_a = rolling_baseline(baseline_a, raw_a)
        base_b = rolling_baseline(baseline_b, raw_b)

        # Calculate deviations from baseline
        dev_a  = raw_a - base_a
        dev_b  = raw_b - base_b

        # Convert to millivolts for display
        mv_a   = counts_to_mv(raw_a)
        mv_b   = counts_to_mv(raw_b)
        dev_mv_a = counts_to_mv(dev_a)
        dev_mv_b = counts_to_mv(dev_b)

        # Send Plant A raw + baseline to WROOM receiver for injection
        # Receiver calculates deviation and sets PWM duty accordingly
        try:
            msg = "{},{}".format(raw_a, base_a)
            en.send(RECEIVER_MAC, msg.encode())
        except Exception as e:
            print("Send error:", e)

        sample += 1

        # Print every 8 samples to avoid flooding serial
        if sample % 8 == 0:
            status = ""

            # Plant A stress detection
            if abs(dev_a) > STRESS_THRESH:
                status += "<<< PLANT A STRESS EVENT  "
            elif abs(dev_a) > STRESS_THRESH // 2:
                status += "<< A signal rising  "

            # Plant B biological response detection
            # This is the interesting one — is Plant B responding
            # with its OWN signal, not just the injected voltage?
            if abs(dev_b) > RESPONSE_THRESH:
                if sample > BASELINE_LEN:
                    status += "*** PLANT B RESPONSE DETECTED"

            print("#{:04d}  {:>8.3f}mV {:>+8.3f}mV  |  {:>8.3f}mV {:>+8.3f}mV  {}".format(
                sample,
                mv_a, dev_mv_a,
                mv_b, dev_mv_b,
                status
            ))

        time.sleep_ms(125)

main()