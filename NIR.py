import time
import board
import busio
import adafruit_as726x

# === Setup I2C for AS7263 ===
i2c = busio.I2C(board.SCL, board.SDA)

# === Initialize AS7263 NIR Sensor ===
as7263 = adafruit_as726x.AS726x_I2C(i2c)

print("🔹 Starting NIR Sensor Reading Loop")

while True:
    try:
        # Turn on LED for illumination
        as7263.driver_led = True

        # Wait until data is ready
        while not as7263.data_ready:
            time.sleep(0.1)

        # Read all 6 NIR spectral channels
        red = as7263.red
        orange = as7263.orange
        yellow = as7263.yellow
        green = as7263.green
        blue = as7263.blue
        violet = as7263.violet
        #18.35116577148437


        # Turn off LED
        as7263.driver_led = False

        # Print readings
        print(f"NIR Readings → Red: {red}, Orange: {orange}, Yellow: {yellow}, Green: {green}, Blue: {blue}, Violet: {violet}")

        # Wait before next reading
        time.sleep(5)  # read every 5 seconds (you can change this)

    except Exception as e:
        print("❌ Error reading NIR sensor:", e)
        time.sleep(2)
