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
import importlib 
import sensor_config 
import calibrate_now

import board, busio
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS1015(i2c)
except Exception:
    print("Hardware warning: I2C or ADS1015 not found.")

# === Setup GPIO for Flasher Light ===
FLASHER_PIN = 16  
GPIO.setmode(GPIO.BCM)
GPIO.setup(FLASHER_PIN, GPIO.OUT)
GPIO.output(FLASHER_PIN, GPIO.LOW) 

# === Configure Gemini API ===
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY environment variable not set. Suggestions will be disabled.")
        gemini_model = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        print("✅ Gemini API configured successfully using gemini-2.5-flash-lite.")

except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    gemini_model = None

# === Import Custom Analysis Modules ===
try:
    from vit_identifier import predict_item
    from main_test_web_banana import run_analysis as run_banana_analysis
    from main_test_web_carrot import run_analysis as run_carrot_analysis
    from main_test_web_mango import run_analysis as run_mango_analysis
    from main_test_web_strawberry import run_analysis as run_strawberry_analysis
    from main_test_web_apple import run_analysis as run_apple_analysis
    from main_test_web_tomato import run_analysis as run_tomato_analysis
    from main_test_web_bellpepper import run_analysis as run_bellpepper_analysis
    from main_test_web_potato import run_analysis as run_potato_analysis
except ImportError as e:
    print(f"❌ Critical Error: Could not import analysis modules.")
    print(f"Make sure vit_identifier.py, etc., are in the same folder.")
    print(f"Error: {e}")

# === Global Hardware Setup ===
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# === Produce Information Database ===
PRODUCE_INFO = {
    "banana": { "name": "Banana", "type": "fruit" },
    "carrot": { "name": "Carrot", "type": "vegetable" },
    "mango": { "name": "Mango", "type": "fruit" },
    "strawberry": { "name": "Strawberry", "type": "fruit" },
    "apple": { "name": "Apple", "type": "fruit" },
    "tomato": { "name": "Tomato", "type": "vegetable" },
    "bellpepper": { "name": "Bellpepper", "type": "vegetable" },
    "potato": { "name": "Potato", "type": "vegetable" },
    "cucumber": { "name": "Cucumber", "type": "vegetable" },
    "orange": { "name": "Orange", "type": "fruit" },
}

# === SENSOR RANGES DATABASE ===
# Format: "ItemName": [Min, Max] for each sensor index [MQ4, MQ135, TGS, NIR]
SENSOR_RANGES = {
    "banana": [
        (20, 24),  # Methane 
        (22, 36),  # Ammonia 
        (13, 17),  # VOCs    
        (80, 34)   # Spectral
    ],
    "carrot": [
        (22, 26),  # Methane 
        (23, 41),  # Ammonia 
        (13, 21),  # VOCs    
        (68, 46)   # Spectral
    ],
    "apple": [
        (21, 25),  # Methane 
        (22, 32),  # Ammonia 
        (13, 17),  # VOCs    
        (110, 30)   # Spectral
    ],
    "cucumber": [
        (21, 25),  # Methane 
        (24, 28),  # Ammonia 
        (13, 15),  # VOCs    
        (51, 44)   # Spectral
    ],
    "mango": [
        (21, 29),  # Methane 
        (23, 56),  # Ammonia 
        (13, 22),  # VOCs    
        (100, 42)   # Spectral
    ],
     "orange": [
        (21, 26),  # Methane 
        (23, 53),  # Ammonia 
        (13, 18),  # VOCs    
        (138, 52)   # Spectral
    ],
     "potato": [
        (21, 24),  # Methane 
        (23, 30),  # Ammonia 
        (13, 16),  # VOCs    
        (75, 40)   # Spectral
    ],
     "tomato": [
        (21, 24),  # Methane 
        (23, 30),  # Ammonia 
        (13, 16),  # VOCs    
        (66, 38)   # Spectral
    ],
     "bellpepper": [
        (21, 24),  # Methane 
        (23, 29),  # Ammonia 
        (13, 16),  # VOCs    
        (104, 50)   # Spectral
    ],
    "strawberry": [
        (21, 26),  # Methane 
        (24, 31),  # Ammonia 
        (13, 16),  # VOCs    
        (70, 33)   # Spectral
    ],
    # Fallback ranges
    "default": [
        (0, 100), (0, 100), (0, 100), (0, 100)
    ]
}

# ==============================
# === Core Logic Functions ===
# ==============================

def capture_vit_image(filename="captured_item.jpg"):
    """Captures an image using fswebcam."""
    try:
        as7263 = adafruit_as726x.AS726x_I2C(i2c)
        as7263.driver_led = True
        GPIO.output(FLASHER_PIN, GPIO.HIGH)
        time.sleep(0.5)
        subprocess.run([
            "fswebcam", "-d", "/dev/video0", "-r", "1280x720", "--no-banner", filename
        ], check=True)
        as7263.driver_led = False
        GPIO.output(FLASHER_PIN, GPIO.LOW)
        print(f"✅ ViT Image captured successfully: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error capturing ViT image: {e}")
        return None

def get_gemini_nutrients(item_name, ripeness_level):
    if not GEMINI_API_KEY:
        return "Nutrient data not available."
    prompt = (
        f"List the main nutrients (e.g., vitamins, minerals) for a '{ripeness_level}' '{item_name}'. "
        f"Be concise, like a comma-separated list. Example: Vitamin C, Potassium, Fiber"
    )
    try:
        response = gemini_model.generate_content(prompt, request_options={'timeout': 30})
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Nutrient data not available."

def fetch_and_update_nutrients(item_name, ripeness_level):
    nutrients = get_gemini_nutrients(item_name, ripeness_level)
    try:
        nutrient_text.config(text=nutrients)
    except tk.TclError:
        pass 

# ========================================================
# === GRAPH UPDATE FUNCTION (RANGE-BASED) ===
# ========================================================
def update_sensor_graph_ui(sensor_vals_list, item_name="default"):
    """
    Updates the horizontal range graph with a Light-to-Dark gradient effect.
    """
    sensor_canvas.delete("all") # Clear previous drawing
    
    # 1. Determine Ranges
    key = item_name.lower()
    if key not in SENSOR_RANGES:
        key = "default"
    ranges = SENSOR_RANGES[key]
    
    # 2. Determine Top Labels
    produce_data = PRODUCE_INFO.get(key, {"type": "fruit"})
    p_type = produce_data.get("type", "fruit")
    
    left_label_text = "Unripe" if p_type == "fruit" else "Fresh"
    right_label_text = "Rotten"
    
    # --- DIMENSIONS ---
    start_y = 35 
    spacing = 35
    bar_height = 25
    
    # Layout Config
    label_end_x = 95   
    bar_start_x = 105   
    bar_width = 240     
    bar_end_x = bar_start_x + bar_width
    
    # Draw Vertical Grid Lines
    sensor_canvas.create_line(bar_start_x, 25, bar_start_x, 160, fill="#dddddd", dash=(2, 2)) 
    sensor_canvas.create_line(bar_end_x, 25, bar_end_x, 160, fill="#dddddd", dash=(2, 2)) 

    # Draw Top Header Labels
    sensor_canvas.create_text(bar_start_x, 15, text=left_label_text, font=("Poppins", 10, "bold"), fill="black", anchor="w")
    sensor_canvas.create_text(bar_end_x, 15, text=right_label_text, font=("Poppins", 10, "bold"), fill="black", anchor="e")

    sensor_labels = ["Freshness", "Spoilage", "Aroma", "Firmness"]
    
    # === GRADIENT COLOR PALETTE (Start Light -> End Dark) ===
    # Format: [Light_Hex, Dark_Hex]
    gradient_colors = [
        ["#B2EBF2", "#00ACC1"], # 0: Methane (Light Cyan -> Dark Cyan)
        ["#C8E6C9", "#43A047"], # 1: Ammonia (Light Green -> Dark Green)
        ["#FFF9C4", "#FDD835"], # 2: VOCs    (Light Yellow -> Dark Gold)
        ["#E1BEE7", "#8E24AA"]  # 3: NIR     (Light Purple -> Dark Purple)
    ]

    # --- HELPER: Hex to RGB conversion ---
    def hex_to_rgb(hex_val):
        hex_val = hex_val.lstrip('#')
        return tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))

    for i in range(4):
        y0 = start_y + (i * spacing)
        y1 = y0 + bar_height
        mid_y = (y0 + y1) / 2
        
        val = sensor_vals_list[i]
        min_val, max_val = ranges[i]
        
        # --- A. Draw Label (Left) ---
        sensor_canvas.create_text(label_end_x, mid_y, text=sensor_labels[i], 
                                  font=("Poppins", 9, "bold"), fill="#333333", anchor="e")
        
        # --- B. Draw Range Bar (GRADIENT EFFECT) ---
        # Get start and end RGB values
        c_start = hex_to_rgb(gradient_colors[i][0])
        c_end = hex_to_rgb(gradient_colors[i][1])
        
        # Draw vertical lines 1px wide to create gradient
        steps = int(bar_width)
        for j in range(steps):
            # Calculate current color (Interpolation)
            r = int(c_start[0] + (c_end[0] - c_start[0]) * (j / steps))
            g = int(c_start[1] + (c_end[1] - c_start[1]) * (j / steps))
            b = int(c_start[2] + (c_end[2] - c_start[2]) * (j / steps))
            color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Draw line
            sensor_canvas.create_line(bar_start_x + j, y0, bar_start_x + j, y1, fill=color)

        # Draw a border around the bar for sharpness
        sensor_canvas.create_rectangle(bar_start_x, y0, bar_end_x, y1, outline="#aaaaaa")

        # --- C. Draw Min/Max Text ---
        if i < 3: 
            min_text = f"{min_val} ppm"
            max_text = f"{max_val} ppm"
        else:
            min_text = str(min_val)
            max_text = str(max_val)

        sensor_canvas.create_text(bar_start_x, y0 - 3, text=min_text, 
                                  font=("Poppins", 6, "bold"), fill="black", anchor="w")
        sensor_canvas.create_text(bar_end_x, y0 - 3, text=max_text, 
                                  font=("Poppins", 6, "bold"), fill="black", anchor="e")

        # --- D. Calculate Indicator Position ---
        range_span = max_val - min_val
        if range_span == 0: range_span = 1
        
        ratio = (val - min_val) / range_span
        
        if ratio < 0: ratio = 0
        if ratio > 1: ratio = 1
        
        pos_x = bar_start_x + (ratio * bar_width)
        
        # --- E. Draw Marker (Red Line) ---\
        sensor_canvas.create_line(pos_x, y0, pos_x, y1, fill="#FF2E2E", width=4)

        # --- F. Draw Current Value Text (ABOVE MARKER) ---
        if i < 3: 
            curr_text = f"{val:.2f}"
        else:
            curr_text = f"{val:.2f}"
            
        # Draw text centered above the red line
        sensor_canvas.create_text(pos_x + 15, y0 + 13, text=curr_text, 
                                  font=("Poppins", 6, "bold"), fill="#000000", anchor="center")

def update_ui(result):
    """Updates all UI elements."""
    item_name = "Unknown"
    pred_name = "Unknown"
    info = PRODUCE_INFO.get("banana") 

    try:
        if not result:
            ripeness_label.config(text="Ripeness Level: Analysis Failed")
            return

        item_name = result.get("item", "Unknown").lower()
        pred_name = result.get("ripeness", "Unknown")
        sensor_values = result.get("sensor_values", [0, 0, 0, 0])
        img_path = result.get("image_path")
        days_to_rotten = result.get("days_to_rotten")
        rotten_date_str = result.get("estimated_rotten_date")

        info = PRODUCE_INFO.get(item_name, PRODUCE_INFO.get("banana")) 
        
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            img_w, img_h = img.size
            target_w, target_h = 268, 135
            aspect = img_w / img_h
            if aspect > (target_w / target_h):
                new_w = target_w; new_h = int(new_w / aspect)
            else:
                new_h = target_h; new_w = int(new_h * aspect)
            
            img = img.resize((new_w, new_h), Image.LANCZOS)
            new_photo = ImageTk.PhotoImage(img)

            image_label.configure(image=new_photo, bg="white")
            image_label.image = new_photo
            image_canvas.delete("image_window")
            image_canvas.create_window(135, 68, window=image_label, anchor="center", tags="image_window")
        
        produce_name_label.config(text=info["name"])
        
        # === PASS item_name TO GRAPH ===
        update_sensor_graph_ui(sensor_values, item_name)
        
        produce_type = info.get("type", "fruit")
        if produce_type == "vegetable":
            label_text = "Freshness Level:"
            ripeness_label.place(x=10, y=335)
            ripeness_label.config(text=f"{label_text} {pred_name}", font=("Poppins", 13, "bold"))
        else:
            label_text = "Ripeness Level:"
            ripeness_label.config(text=f"{label_text} {pred_name}")
        
        draw_gauge(gauge_canvas, pred_name)

        if pred_name in ("Rotten", "Overripe"):
            shelf_life_text.config(text="Item is past its prime.")
        elif days_to_rotten is not None and rotten_date_str is not None:
            shelf_life_text.config(text=f"Approx. {days_to_rotten} days remaining.\n(Est. rotten date: {rotten_date_str})")
        else:
            shelf_life_text.config(text="...")

        nutrient_text.config(text="Getting nutrients...")
        threading.Thread(target=fetch_and_update_nutrients, args=(info["name"],pred_name), daemon=True).start()

    except Exception as e:
        print(f"❌ Error in update_ui: {e}")
        try:
            ripeness_label.config(text="Ripeness Level: Error")
        except Exception:
            pass 

def run_full_analysis_pipeline():
    ripeness_label.config(text="        Identifying..")
    ripeness_label.place(x=75, y=335)
    ripeness_label.config(font=("Poppins", 14, "bold"))
    produce_name_label.config(text="...")
    nutrient_text.config(text="...")
    shelf_life_text.config(text="...") 
    
    # Reset Graph
    update_sensor_graph_ui([0, 0, 0, 0], "default")
    
    draw_gauge(gauge_canvas, "...")
    default_photo = ImageTk.PhotoImage(Image.new("RGB", (1, 1), color="#ffffff"))
    image_label.configure(image=default_photo, bg="white")
    image_label.image = default_photo
    root.update_idletasks()
    
    vit_image_path = capture_vit_image()
    if not vit_image_path:
        ripeness_label.config(text="Ripeness Level: Capture Failed")
        return

    try:
        full_label = predict_item(vit_image_path)
    except Exception as e:
        print(f"❌ ViT prediction error: {e}")
        ripeness_label.config(text="Ripeness Level: ID Failed")
        return

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

    analysis_result = None
    try:
        if item_name == "banana": analysis_result = run_banana_analysis()
        elif item_name == "carrot": analysis_result = run_carrot_analysis()
        elif item_name == "mango": analysis_result = run_mango_analysis()
        elif item_name == "strawberry": analysis_result = run_strawberry_analysis()
        elif item_name == "apple": analysis_result = run_apple_analysis()
        elif item_name == "tomato": analysis_result = run_tomato_analysis()
        elif item_name == "bellpepper": analysis_result = run_bellpepper_analysis()
        elif item_name == "potato": analysis_result = run_potato_analysis()
        else:
            print(f"⚠️ No analysis script found for: {item_name}")
            ripeness_label.config(text=f"No model for {item_name}")
            return
            
    except Exception as e:
        print(f"❌ Error during {item_name} analysis: {e}")
        ripeness_label.config(text="Analysis Error")
        return

    if analysis_result:
        update_ui(analysis_result)
    else:
        ripeness_label.config(text="Analysis Failed")

def run_analysis_thread():
    scan_button.config(state="disabled", text="Scanning...")
    threading.Thread(target=run_analysis_with_button_reset, daemon=True).start()

def run_analysis_with_button_reset():
    try:
        run_full_analysis_pipeline() 
    except Exception as e:
        print(f"Error in analysis thread: {e}")
        try:
            ripeness_label.config(text="Ripeness Level: Error")
        except tk.TclError:
            pass 
    finally:
        try:
            scan_button.config(state="normal", text="Scan Produce")
        except tk.TclError:
            pass 

def run_auto_calibration():
    scan_button.config(text="Calibrating...", bg="#ffc107", state="disabled")
    ripeness_label.config(text="Calibrating Sensors...")
    root.update()
    success = calibrate_now.perform_calibration()
    if success:
        importlib.reload(sensor_config)
        print("🔄 Configuration Reloaded!")
        ripeness_label.config(text="Calibration Done!")
        scan_button.config(text="Scan Produce", bg="#2b8a3e", state="normal")
    else:
        ripeness_label.config(text="Calibration Failed")
        scan_button.config(text="Scan Produce", bg="#2b8a3e", state="normal")

# ==============================
# === GreenMate Tkinter UI ====
# ==============================

root = tk.Tk()
root.title("GreenMate – Freshness Ripeness Simplified")
root.geometry("800x520")
root.configure(bg="#D3F0D8")
root.resizable(False, False)
root.focus_force()
root.attributes('-fullscreen', True) 
root.overrideredirect(True)

def exit_app(event=None):
    print("\n🛑 Escape key pressed. Exiting...")
    root.destroy()
    GPIO.cleanup() 

root.bind('<Escape>', exit_app)

logo_path = "/home/device/ML_model/logo.png"
try:
    logo_img = Image.open(logo_path).resize((120, 60))
    logo_photo = ImageTk.PhotoImage(logo_img)
except Exception as e:
    logo_photo = None

top_frame = tk.Frame(root, bg="#D3F0D8")
top_frame.pack(fill="x", pady=(10, 5))
content_frame = tk.Frame(root, bg="#D3F0D8")
content_frame.pack(expand=True, fill="both")

header_frame = tk.Frame(top_frame, bg="#D3F0D8")
header_frame.pack(side="top", pady=5)
if logo_photo:
    tk.Label(header_frame, image=logo_photo, bg="#D3F0D8").pack(pady=(0,2))
tk.Label(header_frame, text="Eat Fresh. Live Healthy.", bg="#D3F0D8", font=("Poppins", 10, "italic")).pack()

# --- Left Side ---
def create_rounded_rect(canvas, x, y, w, h, r, **kwargs):
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
    canvas.delete("all")
    cx, cy, r = 100, 100, 80
    
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=-60, fill="#28a745", outline="#28a745", style=tk.ARC, width=20) 
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=120, extent=-60, fill="#ffc107", outline="#ffc107", style=tk.ARC, width=20) 
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=60, extent=-60, fill="#dc3545", outline="#dc3545", style=tk.ARC, width=20) 

    if ripeness_name in ("Ripe", "Fresh"): angle_deg = 70 
    elif ripeness_name in ("Unripe", "Underripe", "Not_Fresh"): angle_deg = 30 
    elif ripeness_name in ("Overripe", "Rotten"): angle_deg = 150 
    else: angle_deg = 90 
        
    angle_rad = math.radians(angle_deg)
    lx = cx - (r * 0.85) * math.cos(angle_rad)
    ly = cy - (r * 0.85) * math.sin(angle_rad)
    
    canvas.create_line(cx, cy, lx, ly, fill="black", width=4)
    canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="black", outline="white", width=2)

draw_gauge(gauge_canvas, "—") 

ripeness_label = tk.Label(content_frame, text="  Ripeness Level: —", font=("Poppins", 14, "bold"), bg="#D3F0D8")
ripeness_label.place(x=75, y=335)

# ====================================================
# === RIGHT SIDE (Sensors Graph - RANGE) ===
# ====================================================
sensor_frame = tk.Frame(content_frame, bg="#D3F0D8")
sensor_frame.place(x=380, y=-10)
#tk.Label(sensor_frame, text="Quality Metrics:", font=("Poppins", 12, "bold"), bg="#D3F0D8").pack(anchor="w", pady=(0, 5))

# --- GRAPH SETUP ---
# Reduced width to 400 to fit in the window
graph_width = 480
graph_height = 180 
sensor_canvas = tk.Canvas(sensor_frame, width=graph_width, height=graph_height, bg="#D3F0D8", highlightthickness=0)
sensor_canvas.pack()

# Initial empty draw
update_sensor_graph_ui([0,0,0,0], "default")

# =====================================
# === RIGHT SIDE (Panel & Button) ===
# =====================================

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
panel_height = 210 
right_panel_canvas = tk.Canvas(content_frame, bg="#D3F0D8", bd=0, highlightthickness=0)
right_panel_canvas.place(x=380, y=170, width=panel_width, height=panel_height)
create_rounded_panel(right_panel_canvas, 0, 0, panel_width, panel_height, 20)
inner_panel = tk.Frame(right_panel_canvas, bg="#ffffff")
right_panel_canvas.create_window(1, 1, window=inner_panel, anchor="nw", 
                                width=panel_width-2, height=panel_height-2)

nutrient_label = tk.Label(inner_panel, text="Nutritients:", font=("Poppins", 12, "bold"), bg="#ffffff")
nutrient_label.pack(side="top", anchor="w", padx=20, pady=(15,0))
nutrient_text = tk.Label(inner_panel, text="", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
nutrient_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")

shelf_life_label = tk.Label(inner_panel, text="Shelf Life:", font=("Poppins", 12, "bold"), bg="#ffffff")
shelf_life_label.pack(side="top", anchor="w", padx=20, pady=(10,0)) 
shelf_life_text = tk.Label(inner_panel, text="...", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
shelf_life_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")

try:
    icon_path = "/home/device/ML_model/scan_icon.png"
    scan_icon_img = Image.open(icon_path).resize((20, 20))
    scan_icon_photo = ImageTk.PhotoImage(scan_icon_img)
    btn_compound = "left"
    btn_image = scan_icon_photo
    btn_padx = 5
except Exception as e:
    scan_icon_photo = None
    btn_compound = "none"
    btn_image = None
    btn_padx = 0

button_frame = tk.Frame(inner_panel, bg="#ffffff")
button_frame.pack(side="bottom", fill="x", pady=10)
scan_button = tk.Button(button_frame, text="Scan Produce", font=("Poppins", 10, "bold"),
                        bg="#2b8a3e", fg="white", relief="raised", 
                        image=btn_image, compound=btn_compound,
                        padx=btn_padx,
                        pady=2,
                        command=run_analysis_thread)
scan_button.pack(pady=0)
if scan_icon_photo:
    scan_button.image = scan_icon_photo

# ==============================
# === GPIO Check Loop =====
# ==============================

def check_button():
    try:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            start_time = time.time()
            long_press_triggered = False

            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.1)
                if not long_press_triggered and (time.time() - start_time > 3):
                    print("🔘 Long Press: Calibrating...")
                    run_auto_calibration()
                    long_press_triggered = True 

            if not long_press_triggered:
                if scan_button["state"] == "normal":
                    print("🔘 Short Press: Scanning...")
                    run_analysis_thread()

        root.after(100, check_button)

    except tk.TclError:
        pass 

print("\n🔹 System ready. Launching GreenMate UI...\n")
root.after(300, check_button)
root.mainloop()

print("🛑 Exiting... cleaning up GPIO.")
GPIO.cleanup()