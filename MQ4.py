import time
import board
import busio
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
import sensor_config

# Setup I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize ADC
ads = ADS1015(i2c)

# Channel 0 (A0 on ADS1015)
chan = AnalogIn(ads, 0)

print("Reading MQ4 sensor values...")

while True:
    voltage = chan.voltage-sensor_config.OFFSETS["MQ4"]
    print(f"MQ4 Voltage: {voltage:.3f} V")
    time.sleep(2)

#0.10