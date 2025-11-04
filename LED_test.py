import RPi.GPIO as GPIO
import time

# --- Configuration ---
LED_PIN = 16  # This is GPIO16 (BCM numbering)

# --- Setup GPIO ---
GPIO.setmode(GPIO.BCM)              # Use BCM pin numbering
GPIO.setup(LED_PIN, GPIO.OUT)       # Set the pin as an output
GPIO.output(LED_PIN, GPIO.LOW)      # Ensure LED is off to start

print(f"Testing LED on GPIO {LED_PIN}...")
print("Press CTRL+C to stop the test at any time.")

try:
    # Turn LED ON
    print("Turning LED ON...")
    GPIO.output(LED_PIN, GPIO.HIGH)
    
    # Wait for 3 seconds
    time.sleep(3)
    
    # Turn LED OFF
    print("Turning LED OFF...")
    GPIO.output(LED_PIN, GPIO.LOW)
    
    print("\nTest complete.")

except KeyboardInterrupt:
    print("Test stopped by user.")

finally:
    # This part always runs, even if you interrupt the script
    print("Cleaning up GPIO pins.")
    GPIO.cleanup()