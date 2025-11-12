import time
import board
import busio
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
from Adafruit_IO import Client
import adafruit_as726x
import os

ADAFRUIT_IO_USERNAME1 = os.environ.get("ADAFRUIT_IO_USERNAME_Orange_Apple_Bellpepper")
ADAFRUIT_IO_KEY1 = os.environ.get("ADAFRUIT_IO_KEY_Orange_Apple_Bellpepper")


# === Initialize Adafruit IO Client ===
aio1 = Client(ADAFRUIT_IO_USERNAME1, ADAFRUIT_IO_KEY1)

# === Setup I2C for both ADS1015 and AS7263 ===
i2c = busio.I2C(board.SCL, board.SDA)

# === Gas Sensors (ADS1015) Setup ===
ads = ADS1015(i2c)
mq4_chan = AnalogIn(ads, 0)       # A0 for MQ4
mq135_chan = AnalogIn(ads, 1)     # A1 for MQ135
tgs2602_chan = AnalogIn(ads, 2)   # A2 for TGS2602

# === Sample ID ===
sample_id = "cucumber3"  # <-- Change this for each banana (banana1, banana2, etc.)

print("Starting sensor loop for", sample_id)

# === NIR Read Timer Setup ===
last_nir_time = time.time()
nir_interval = 200  # 5 minutes in seconds

while True:
    try:
        # === GAS SENSOR SECTION (every 1 minute) ===
        mq4_voltage = mq4_chan.voltage-0.010
        print(f"[{sample_id}] MQ4 Voltage: {mq4_voltage:.3f} V")
        aio1.send('mq4-gas-cucumber', f"{sample_id}:{mq4_voltage}")
        time.sleep(0.5)

        mq135_voltage = mq135_chan.voltage-0.078
        print(f"[{sample_id}] MQ135 Voltage: {mq135_voltage:.3f} V")
        aio1.send('mq135-gas-cucumber', f"{sample_id}:{mq135_voltage}")
        time.sleep(0.5)

        tgs2602_voltage = tgs2602_chan.voltage
        print(f"[{sample_id}] TGS2602 Voltage: {tgs2602_voltage:.3f} V")
        aio1.send('tgs2602-gas-cucumber', f"{sample_id}:{tgs2602_voltage}")

        # === NIR SENSOR SECTION (every 5 minutes) ===
        current_time = time.time()
        if current_time - last_nir_time >= nir_interval:
            print(f"[{sample_id}] Reading AS7263 NIR channels...")

            time.sleep(1)
            as7263 = adafruit_as726x.AS726x_I2C(i2c)
            as7263.driver_led = True

            while not as7263.data_ready:
                time.sleep(0.1)

            red = as7263.red
            orange = as7263.orange
            yellow = as7263.yellow
            green = as7263.green
            blue = as7263.blue
            violet = as7263.violet+30.35116577148437
            as7263.driver_led = False

            nir_data = f"{sample_id}:{red},{orange},{yellow},{green},{blue},{violet}"
            aio1.send('cucumber-nir', nir_data)
            print(f"[{sample_id}] NIR Sent: {nir_data}")

            last_nir_time = current_time

    except Exception as e:
        print(f"[{sample_id}] Error:", e)

    time.sleep(60)  # Wait 1 minute before next gas sensor read
