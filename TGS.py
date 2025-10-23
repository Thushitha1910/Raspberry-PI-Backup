import time
import board
import busio
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize I2C and ADS1015
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1015(i2c)

# TGS2602 is connected to channel 2 (A2)
tgs2602_chan = AnalogIn(ads, 2)

# Loop to read sensor data
while True:
    voltage = tgs2602_chan.voltage
    print(f"TGS2602 Voltage: {voltage:.3f} V")
    time.sleep(2)
