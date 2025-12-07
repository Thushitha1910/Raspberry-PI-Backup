import time
import board
import busio
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
import sensor_config

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1015(i2c)
mq135_chan = AnalogIn(ads, 1)  # Only channel 1

while True:
    voltage = mq135_chan.voltage-sensor_config.OFFSETS["MQ135"]
    print(f"MQ135 Voltage: {voltage:.3f} V")
    time.sleep(2)

#0.078
#0.108
#0.198