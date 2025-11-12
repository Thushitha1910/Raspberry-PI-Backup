# =======================================================================
# === GreenMate Main UI Application ===
# =======================================================================
# This script is the new main entry point for the application.
# It integrates the Tkinter UI with the multi-fruit detection logic.
# =======================================================================

import time
import board
import busio
import torch
import torch.nn as nn
import joblib
import numpy as np
import RPi.GPIO as GPIO
from PIL import Image, ImageTk
import google.generativeai as genai
import math
import subprocess
import cv2
import os
import tkinter as tk
from tkinter import ttk
import threading
import adafruit_as726x
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
import RPi.GPIO as GPIO

import board, busio
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1015(i2c)

# === NEW: Setup GPIO for Flasher Light ===
FLASHER_PIN = 16  # This is GPIO16 (BCM naming)
GPIO.setmode(GPIO.BCM)
GPIO.setup(FLASHER_PIN, GPIO.OUT)
GPIO.output(FLASHER_PIN, GPIO.LOW) # Ensure it's off to start

# === NEW: Configure Gemini API ===
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY environment variable not set. Suggestions will be disabled.")
        gemini_model = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        print("✅ Gemini API configured successfully using gemini-2.5-flash.")

except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    gemini_model = None

# === Import Custom Analysis Modules ===
# (Make sure these files are in the same directory or Python path)
try:
    from vit_identifier import predict_item
    from main_test_web_banana import run_analysis as run_banana_analysis
    from main_test_web_carrot import run_analysis as run_carrot_analysis
    from main_test_web_mango import run_analysis as run_mango_analysis
    from main_test_web_strawberry import run_analysis as run_strawberry_analysis
    from main_test_web_apple import run_analysis as run_apple_analysis
    from main_test_web_tomato import run_analysis as run_tomato_analysis
    from main_test_web_bellpepper import run_analysis as run_bellpepper_analysis
    # ... import other analysis scripts here ...
except ImportError as e:
    print(f"❌ Critical Error: Could not import analysis modules.")
    print(f"Make sure vit_identifier.py, main_test_web_banana.py, etc., are in the same folder.")
    print(f"Error: {e}")
    exit()


# === Global Hardware Setup ===
# (No need to set this up in the other scripts anymore)
try:
    i2c = busio.I2C(board.SCL, board.SDA)
except RuntimeError:
    print("⚠️ Could not initialize I2C. Hardware may not be connected.")
    # You might want to exit() or have a 'demo mode'
    
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# === Produce Information Database ===
# Central place to store UI text for each item
PRODUCE_INFO = {
    "banana": {
        "name": "Banana",
        "type": "fruit"
    },
    "carrot": {
        "name": "Carrot",
        "type": "vegetable"
    },
    "mango": {
        "name": "Mango",
        "type": "fruit"
    },
    "strawberry": {
        "name": "Strawberry",
        "type": "fruit"
    },
    "apple": {
        "name": "Apple",
        "type": "fruit"
    },
    "tomato": {
        "name": "Tomato",
        "type": "vegetable"  # Botanically a fruit, and "ripeness" applies
    },
    "bellpepper": {
        "name": "Bellpepper",
        "type": "vegetable"  # Botanically a fruit, and "ripeness" applies
    },
}

# ==============================
# === Core Logic Functions ===
# ==============================

def capture_vit_image(filename="captured_item.jpg"):
    """
    Captures an image using fswebcam for the ViT identifier.
    This is the simple capture from main_fruit_detector.py.
    """
    try:
        as7263 = adafruit_as726x.AS726x_I2C(i2c)
        as7263.driver_led = True
        GPIO.output(FLASHER_PIN, GPIO.HIGH)
        time.sleep(0.5)
        subprocess.run([
            "fswebcam",
            "-d", "/dev/video0",
            "-r", "1280x720",
            "--no-banner",
            filename
        ], check=True)
        as7263.driver_led = False
        GPIO.output(FLASHER_PIN, GPIO.LOW)
        print(f"✅ ViT Image captured successfully: {filename}")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"❌ Error capturing ViT image: {e}")
        return None

def get_gemini_nutrients(item_name,ripeness_level):
    """
    Gets a dynamic nutrient list from the Gemini API.
    """
    if not GEMINI_API_KEY:
        # Fallback if API key is not set
        return "Nutrient data not available."

    prompt = (
        f"List the main nutrients (e.g., vitamins, minerals) for a '{ripeness_level}' '{item_name}'. "
        f"Be concise, like a comma-separated list. "
        f"Example: Vitamin C, Potassium, Fiber"
    )
    
    try:
        # Set a 10-second timeout for the network request
        response = gemini_model.generate_content(
            prompt,
            request_options={'timeout': 30}
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Nutrient data not available."


def fetch_and_update_nutrients(item_name,ripeness_level):
    """
    Runs the Gemini query in a thread and updates the UI when complete.
    """
    # Get the dynamic nutrients
    nutrients = get_gemini_nutrients(item_name,ripeness_level)
    
    # Update the UI label (must be thread-safe)
    try:
        # This is safe because Tkinter updates are atomic
        nutrient_text.config(text=nutrients)
    except tk.TclError:
        pass # Window was closed before the thread finished
        
def update_ui(result):
    """
    Updates all UI elements based on the final result dictionary.
    (CORRECTED VERSION)
    """
    # === Initialize variables to safe defaults ===
    item_name = "Unknown"
    pred_name = "Unknown"
    info = PRODUCE_INFO.get("banana") # Default to banana info

    try:
        if not result:
            ripeness_label.config(text="Ripeness Level: Analysis Failed")
            return

        # === 1. Get data from result dictionary ===
        item_name = result.get("item", "Unknown").lower()
        pred_name = result.get("ripeness", "Unknown")
        sensor_values = result.get("sensor_values", [0, 0, 0, 0])
        img_path = result.get("image_path")
        days_to_rotten = result.get("days_to_rotten")
        rotten_date_str = result.get("estimated_rotten_date")

        # === 2. Get the produce info (THIS IS THE FIX) ===
        #    Assign 'info' *after* 'item_name' is known, and *before* 'info' is used.
        info = PRODUCE_INFO.get(item_name, PRODUCE_INFO.get("banana")) 
        
        # === 3. Update Image ===
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            img_w, img_h = img.size
            target_w, target_h = 268, 135
            aspect = img_w / img_h
            if aspect > (target_w / target_h):
                new_w = target_w
                new_h = int(new_w / aspect)
            else:
                new_h = target_h
                new_w = int(new_h * aspect)
            
            img = img.resize((new_w, new_h), Image.LANCZOS)
            new_photo = ImageTk.PhotoImage(img)

            image_label.configure(image=new_photo, bg="white")
            image_label.image = new_photo
            image_canvas.delete("image_window")
            image_canvas.create_window(135, 68, window=image_label, anchor="center", tags="image_window")
        else:
            print(f"⚠️ Could not load image from path: {img_path}")
        
        # === 4. Update all text labels ===
        produce_name_label.config(text=info["name"])
        
        sensor_labels["MQ 4 Sensor"].config(text=f"{sensor_values[0]:.2f} ppm")
        sensor_labels["MQ 135 Sensor"].config(text=f"{sensor_values[1]:.2f} ppm")
        sensor_labels["TGS2602 Sensor"].config(text=f"{sensor_values[2]:.2f} ppm")
        sensor_labels["NIR Spectrometer"].config(text=f"{sensor_values[3]:.2f}")
        
        # === NEW: Dynamically set label based on produce type ===
        produce_type = info.get("type", "fruit") # Default to 'fruit' if not specified
        
        if produce_type == "vegetable":
            label_text = "Freshness Level:"
            ripeness_label.place(x=10, y=335)
            ripeness_label.config(
                text=f"{label_text} {pred_name}", 
                font=("Poppins", 13, "bold")  # <-- SET NEW FONT
            )
        else:
            label_text = "Ripeness Level:"
            
        ripeness_label.config(text=f"{label_text} {pred_name}")
        # === END NEW ===
        
        draw_gauge(gauge_canvas, pred_name) # Update the gauge

        # === 5. Update Shelf Life ===
        if pred_name in ("Rotten", "Overripe"):
            shelf_life_text.config(text="Item is past its prime.")
        elif days_to_rotten is not None and rotten_date_str is not None:
            shelf_life_text.config(text=f"Approx. {days_to_rotten} days remaining.\n(Est. rotten date: {rotten_date_str})")
        else:
            shelf_life_text.config(text="...")

        # === 6. Update Nutrients (dynamically) ===
        nutrient_text.config(text="Getting nutrients...")
        threading.Thread(
            target=fetch_and_update_nutrients, 
            args=(info["name"],pred_name), # Now 'info' is guaranteed to exist
            daemon=True
        ).start()

    except Exception as e:
        # If ANY error happens, print it and show a user-friendly message
        print(f"❌ Error in update_ui: {e}")
        try:
            ripeness_label.config(text="Ripeness Level: Error")
            produce_name_label.config(text="Error")
            nutrient_text.config(text="...")
            shelf_life_text.config(text="...")
        except Exception:
            pass # UI might already be closed


def run_full_analysis_pipeline():
    """
    This is the main function that orchestrates the entire process.
    """
    # --- 1. Clear UI for new scan ---
    ripeness_label.config(text="        Identifying..")
    ripeness_label.place(x=75, y=335)
    ripeness_label.config(font=("Poppins", 14, "bold"))
    produce_name_label.config(text="...")
    nutrient_text.config(text="...")
    #suggestion_text.config(text="...")
    shelf_life_text.config(text="...") # === NEW ===
    sensor_labels["MQ 4 Sensor"].config(text="... ppm")
    sensor_labels["MQ 135 Sensor"].config(text="... ppm")
    sensor_labels["TGS2602 Sensor"].config(text="... ppm")
    sensor_labels["NIR Spectrometer"].config(text="...")
    draw_gauge(gauge_canvas, "...")
    default_photo = ImageTk.PhotoImage(Image.new("RGB", (1, 1), color="#ffffff"))
    image_label.configure(image=default_photo, bg="white")
    image_label.image = default_photo
    root.update_idletasks()
    
    # --- 2. Capture image for ViT ---
    vit_image_path = capture_vit_image()
    if not vit_image_path:
        ripeness_label.config(text="Ripeness Level: Capture Failed")
        return

    # --- 3. Identify item with ViT ---
    try:
        full_label = predict_item(vit_image_path)
    except Exception as e:
        print(f"❌ ViT prediction error: {e}")
        ripeness_label.config(text="Ripeness Level: ID Failed")
        return

    # --- 4. Determine generic item name ---
    fruits_prefixes = ["ripe", "unripe", "rotten"]
    item_name = ""
    if any(full_label.lower().startswith(fk) for fk in fruits_prefixes):
        item_name = full_label.split("_")[1].lower()
    else:
        item_name = full_label.split("_")[0].lower()
    
    print(f"🧠 Identified item: {item_name}")
    ripeness_label.config(text=f"Analyzing {item_name.capitalize()}...")
    produce_name_label.config(text=item_name.capitalize())
    root.update_idletasks()

    # --- 5. Run specific analysis script ---
    analysis_result = None
    try:
        if item_name == "banana":
            analysis_result = run_banana_analysis()
        elif item_name == "carrot":
            analysis_result = run_carrot_analysis()
        elif item_name == "mango":
            analysis_result = run_mango_analysis()
        elif item_name == "strawberry":
            analysis_result = run_strawberry_analysis()
        elif item_name == "apple":
            analysis_result = run_apple_analysis()
        elif item_name == "tomato":
            analysis_result = run_tomato_analysis()
        elif item_name == "bellpepper":
            analysis_result = run_bellpepper_analysis()
        else:
            print(f"⚠️ No analysis script found for: {item_name}")
            ripeness_label.config(text=f"No model for {item_name}")
            return
            
    except Exception as e:
        print(f"❌ Error during {item_name} analysis: {e}")
        ripeness_label.config(text="Analysis Error")
        return

    # --- 6. Update UI with final results ---
    if analysis_result:
        update_ui(analysis_result)
    else:
        ripeness_label.config(text="Analysis Failed")


# --- Run in Thread (so UI doesn’t freeze) ---
def run_analysis_thread():
    """Starts the analysis pipeline in a separate thread."""
    scan_button.config(state="disabled", text="Scanning...")
    threading.Thread(target=run_analysis_with_button_reset, daemon=True).start()

def run_analysis_with_button_reset():
    """Wraps analysis to re-enable button after completion."""
    try:
        run_full_analysis_pipeline() # <-- This is the new main function
    except Exception as e:
        print(f"Error in analysis thread: {e}")
        try:
            ripeness_label.config(text="Ripeness Level: Error")
        except tk.TclError:
            pass # UI already closed
    finally:
        # Re-enable button
        try:
            scan_button.config(state="normal", text="Scan Produce")
        except tk.TclError:
            pass # UI already closed


# ==============================
# === GreenMate Tkinter UI ====
# ==============================
# (This is all the same UI code from your original script)

# --- Tkinter Setup ---
root = tk.Tk()
root.title("GreenMate – Freshness Ripeness Simplified")
root.geometry("800x480")
root.configure(bg="#D3F0D8")
root.resizable(False, False)

# --- Load Logo ---
logo_path = "/home/device/ML_model/logo.png"
try:
    logo_img = Image.open(logo_path).resize((120, 60))
    logo_photo = ImageTk.PhotoImage(logo_img)
except Exception as e:
    print(f"Logo load error: {e}")
    logo_photo = None

# --- Layout Frames ---
top_frame = tk.Frame(root, bg="#D3F0D8")
top_frame.pack(fill="x", pady=(10, 5))
content_frame = tk.Frame(root, bg="#D3F0D8")
content_frame.pack(expand=True, fill="both")

# --- Header ---
header_frame = tk.Frame(top_frame, bg="#D3F0D8")
header_frame.pack(side="top", pady=5)
if logo_photo:
    tk.Label(header_frame, image=logo_photo, bg="#D3F0D8").pack(pady=(0,2))
tk.Label(header_frame, text="Freshness Ripeness Simplified", bg="#D3F0D8", font=("Poppins", 9, "normal")).pack()

# ==================================
# === LEFT SIDE (Image & Gauge) ====
# ==================================

def create_rounded_rect(canvas, x, y, w, h, r, **kwargs):
    # === FIXED: Typo 9NT changed to 90 ===
    canvas.create_arc(x, y, x+2*r, y+2*r, start=90, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x+w-2*r, y, x+w, y+2*r, start=0, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x, y+h-2*r, x+2*r, y+h, start=180, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x+w-2*r, y+h-2*r, x+w, y+h, start=270, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_rectangle(x+r, y, x+w-r, y+h, **kwargs)
    canvas.create_rectangle(x, y+r, x+w, y+h-r, **kwargs)

image_canvas_width = 270
image_canvas_height = 137
image_canvas = tk.Canvas(content_frame, width=image_canvas_width, height=image_canvas_height, bg="#D3F0D8", highlightthickness=0)
image_canvas.place(x=50, y=20)
create_rounded_rect(image_canvas, 0, 0, image_canvas_width, image_canvas_height, 15, fill="white", outline="")
default_img = Image.new("RGB", (1, 1), color="#ffffff")
default_photo = ImageTk.PhotoImage(default_img)
image_label = tk.Label(image_canvas, image=default_photo, bg="white")
image_canvas.create_window(image_canvas_width/2, image_canvas_height/2, window=image_label, anchor="center", tags="image_window")

produce_name_label = tk.Label(content_frame, text="...", font=("Poppins", 16, "bold"), bg="#D3F0D8")
produce_name_label.place(x=185, y=162, anchor="n") 

gauge_canvas = tk.Canvas(content_frame, width=200, height=120, bg="#D3F0D8", highlightthickness=0)
gauge_canvas.place(x=85, y=210)

def draw_gauge(canvas, ripeness_name):
    """Draws the ripeness gauge. Now handles multiple item labels."""
    canvas.delete("all")
    cx, cy, r = 100, 100, 80
    
    # Draw background arcs
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=-60, 
                        fill="#28a745", outline="#28a745", style=tk.ARC, width=20) # Green
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=120, extent=-60, 
                        fill="#ffc107", outline="#ffc107", style=tk.ARC, width=20) # Yellow
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=60, extent=-60, 
                        fill="#dc3545", outline="#dc3545", style=tk.ARC, width=20) # Red

    # === UPDATED MAPPING ===
    # Map ripeness name to angle
    if ripeness_name in ("Ripe", "Fresh"):
        angle_deg = 70 # Pointing at green
    elif ripeness_name in ("Unripe", "Underripe", "Not_Fresh"):
        angle_deg = 30 # Pointing at yellow
    elif ripeness_name in ("Overripe", "Rotten"):
        angle_deg = 150 # Pointing at red
    else: # Default/unknown state
        angle_deg = 90 # Pointing straight up (middle)
        
    angle_rad = math.radians(angle_deg)
    lx = cx - (r * 0.85) * math.cos(angle_rad)
    ly = cy - (r * 0.85) * math.sin(angle_rad)
    
    canvas.create_line(cx, cy, lx, ly, fill="black", width=4)
    canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="black", outline="white", width=2)

draw_gauge(gauge_canvas, "—") 

ripeness_label = tk.Label(content_frame, text="  Ripeness Level: —", font=("Poppins", 14, "bold"), bg="#D3F0D8")
ripeness_label.place(x=75, y=335)

# =====================================
# === RIGHT SIDE (Sensors & Panel) ===
# =====================================

sensor_frame = tk.Frame(content_frame, bg="#D3F0D8")
sensor_frame.place(x=380, y=20)
tk.Label(sensor_frame, text="Sensors in active:", font=("Poppins", 12, "bold"), bg="#D3F0D8").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
sensor_labels = {}
for i, name in enumerate(["MQ 4 Sensor", "MQ 135 Sensor", "TGS2602 Sensor", "NIR Spectrometer"], start=1):
    tk.Label(sensor_frame, text=f"{name}:", font=("Poppins", 12, "normal"), bg="#D3F0D8").grid(row=i, column=0, sticky="w", padx=(0, 20))
    lbl = tk.Label(sensor_frame, text="—", font=("Poppins", 12, "bold"), bg="#D3F0D8", width=10, anchor="e")
    lbl.grid(row=i, column=1, padx=10, sticky="e")
    sensor_labels[name] = lbl

def create_rounded_panel(canvas, x, y, w, h, r):
    border_color = "#bbbbbb"
    fill_color = "#ffffff"
    canvas.create_rectangle(x, y+r, x+w, y+h, fill=fill_color, outline="")
    canvas.create_rectangle(x+r, y, x+w-r, y+h, fill=fill_color, outline="")
    canvas.create_arc(x, y, x+2*r, y+2*r, start=90, extent=90, fill=fill_color, outline="", style=tk.PIESLICE)
    canvas.create_arc(x+w-2*r, y, x+w, y+2*r, start=0, extent=90, fill=fill_color, outline="", style=tk.PIESLICE)
    canvas.create_arc(x, y, x+2*r, y+2*r, start=90, extent=90, style=tk.ARC, outline=border_color, width=1)
    canvas.create_arc(x+w-2*r, y, x+w-1, y+2*r, start=0, extent=90, style=tk.ARC, outline=border_color, width=1)
    canvas.create_line(x+r, y, x+w-r, y, fill=border_color, width=1)
    canvas.create_line(x, y+r, x, y+h-1, fill=border_color, width=1)
    canvas.create_line(x+w-1, y+r, x+w-1, y+h-1, fill=border_color, width=1)
    canvas.create_line(x, y+h-1, x+w-1, y+h-1, fill=border_color, width=1)

panel_width = 400
# === FIXED: Increased panel height to fit button ===
panel_height = 210 # Was 230
right_panel_canvas = tk.Canvas(content_frame, bg="#D3F0D8", bd=0, highlightthickness=0)
right_panel_canvas.place(x=380, y=155, width=panel_width, height=panel_height)
create_rounded_panel(right_panel_canvas, 0, 0, panel_width, panel_height, 20)
inner_panel = tk.Frame(right_panel_canvas, bg="#ffffff")
right_panel_canvas.create_window(1, 1, window=inner_panel, anchor="nw", 
                                width=panel_width-2, height=panel_height-2)

nutrient_label = tk.Label(inner_panel, text="Nutritients:", font=("Poppins", 12, "bold"), bg="#ffffff")
nutrient_label.pack(side="top", anchor="w", padx=20, pady=(15,0))
nutrient_text = tk.Label(inner_panel, text="", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
nutrient_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")

# suggestion_label = tk.Label(inner_panel, text="Suggestions:", font=("Poppins", 12, "bold"), bg="#ffffff")
# suggestion_label.pack(side="top", anchor="w", padx=20, pady=(15,0))
# suggestion_text = tk.Label(inner_panel, text="", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
# suggestion_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")

# === NEW: Shelf Life Labels ===
shelf_life_label = tk.Label(inner_panel, text="Shelf Life:", font=("Poppins", 12, "bold"), bg="#ffffff")
shelf_life_label.pack(side="top", anchor="w", padx=20, pady=(10,0)) # Reduced top padding slightly
shelf_life_text = tk.Label(inner_panel, text="...", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
shelf_life_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")
# === END NEW ===


try:
    icon_path = "/home/device/ML_model/scan_icon.png"
    scan_icon_img = Image.open(icon_path).resize((30, 30))
    scan_icon_photo = ImageTk.PhotoImage(scan_icon_img)
    btn_compound = "left"
    btn_image = scan_icon_photo
    btn_padx = 10
except Exception as e:
    print(f"Scan icon load error: {e}. Using text-only button.")
    scan_icon_photo = None
    btn_compound = "none"
    btn_image = None
    btn_padx = 0

button_frame = tk.Frame(inner_panel, bg="#ffffff")
button_frame.pack(side="bottom", fill="x", pady=13)
scan_button = tk.Button(button_frame, text="Scan Produce", font=("Poppins", 13, "bold"),
                        bg="#2b8a3e", fg="white", relief="raised", 
                        image=btn_image, compound=btn_compound,
                        padx=btn_padx,
                        command=run_analysis_thread) # This now calls the new pipeline
scan_button.pack(pady=0)
if scan_icon_photo:
    scan_button.image = scan_icon_photo

# ==============================
# === GPIO Check Loop =====
# ==============================

def check_button():
    try:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            if scan_button["state"] == "normal":
                print("🔘 Physical button pressed!")
                run_analysis_thread() # This also calls the new pipeline
        root.after(300, check_button)
    except tk.TclError:
        pass # Window was closed

print("\n🔹 System ready. Launching GreenMate UI...\n")
root.after(300, check_button)
root.mainloop()

# Clean up GPIO on exit
print("🛑 Exiting... cleaning up GPIO.")
GPIO.cleanup()