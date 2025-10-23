import RPi.GPIO as GPIO
import subprocess
import time
from vit_identifier import predict_item  # ViT API prediction function

# === GPIO SETUP ===
BUTTON_PIN = 17  # Change if you use a different GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# === Image capture function ===
def capture_image(filename="captured_item.jpg"):
    try:
        subprocess.run([
            "fswebcam",
            "-d", "/dev/video0",
            "-r", "1280x720",
            "--no-banner",
            filename
        ], check=True)
        print(f"✅ Image captured successfully: {filename}")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"❌ Error capturing image: {e}")
        return None

# === Run corresponding analysis script ===
def run_analysis_script(item_name):
    scripts = {
        "banana": "main_test_web_banana.py",
        #"mango": "main_test_web_mango.py",
        #"apple": "main_test_web_apple.py",
        #"orange": "main_test_web_orange.py",
        #"strawberry": "main_test_web_strawberry.py",
        "carrot": "main_test_web_carrot.py",
        #"potato": "main_test_web_potato.py",
        #"tomato": "main_test_web_tomato.py",
        #"bellpepper": "main_test_web_bellpepper.py",
        #"cucumber": "main_test_web_cucumber.py",
    }

    if item_name.lower() in scripts:
        script_path = f"/home/device/ML_model/{scripts[item_name.lower()]}"
        print(f"➡️ Launching analysis for {item_name} using {script_path}...")
        subprocess.run(["python3", script_path])
    else:
        print(f"⚠️ No model found for detected item: {item_name}")

# === Main detection pipeline ===
def run_detection_pipeline():
    print("\n🔹 Starting item identification process...\n")

    # Step 1: Capture image
    filename = capture_image()
    if not filename:
        print("❌ Image capture failed. Try again.")
        return

    # Step 2: Identify fruit/vegetable via ViT API
    full_label = predict_item(filename)  # This returns label like 'ripe_banana' or 'Carrot_fresh'
    
    # Step 3: Determine generic item name
    fruits_prefixes = ["ripe", "unripe", "rotten"]  # prefixes for fruits
    if any(full_label.lower().startswith(fk) for fk in fruits_prefixes):
        # Fruits: take second word after underscore
        item_name = full_label.split("_")[1].lower()
    else:
        # Vegetables: take first word before underscore (ignore freshness/rot)
        item_name = full_label.split("_")[0].lower()

    print(f"🧠 Identified item for analysis: {item_name}")

    # Step 4: Run specific script
    run_analysis_script(item_name)
    print("✅ Detection cycle complete.\n")

# === Button trigger loop ===
print("🍏 System ready — press the button to detect fruit/vegetable.\n")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("🔘 Button pressed! Running detection pipeline...")
            run_detection_pipeline()
            time.sleep(2)  # debounce delay
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Exiting program...")
    GPIO.cleanup()
