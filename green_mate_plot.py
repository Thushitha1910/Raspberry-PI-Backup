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
import importlib  # <--- NEW
import sensor_config # <--- Make sure this is imported
import calibrate_now

import board, busio
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS1015(i2c)
except Exception:
    print("Hardware warning: I2C or ADS1015 not found.")

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
except ImportError as e:
    print(f"❌ Critical Error: Could not import analysis modules.")
    print(f"Make sure vit_identifier.py, etc., are in the same folder.")
    print(f"Error: {e}")
    # exit() # Commented out to allow UI testing without modules

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
    """Gets a dynamic nutrient list from the Gemini API."""
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
    """Runs the Gemini query in a thread and updates the UI."""
    nutrients = get_gemini_nutrients(item_name, ripeness_level)
    try:
        nutrient_text.config(text=nutrients)
    except tk.TclError:
        pass 

# ========================================================
# === GRAPH UPDATE FUNCTION (HORIZONTAL) ===
# ========================================================
def update_sensor_graph_ui(sensor_vals_list):
    """
    Updates the horizontal bar graph width and values.
    Expects a list: [MQ4_val, MQ135_val, TGS_val, NIR_val]
    """
    # Define max values to calculate percentage width
    max_values = [60, 60, 60, 150] 
    sensor_names = ["MQ4", "MQ135", "TGS", "NIR"]
    
    # Define colors
    bar_colors = ["#17a2b8", "#28a745", "#ffc107", "#6f42c1"]
    
    # Configuration for Horizontal Graph (Must match Setup below)
    start_x = 105        # Where the bars start (to leave room for text on left)
    max_bar_width = 210  # Maximum length of a bar in pixels
    
    for i, val in enumerate(sensor_vals_list):
        # 1. Calculate ratio
        ratio = min(val / max_values[i], 1.0) # Cap at 100%
        
        # 2. Calculate pixel width
        px_width = ratio * max_bar_width
        
        # 3. Get tag names
        tag_name = sensor_names[i]
        val_tag_name = tag_name + "_val"

        try:
            # coords: x0, y0, x1, y1
            current_coords = sensor_canvas.coords(tag_name)
            if current_coords:
                # Keep y0 and y1 the same (vertical position)
                # Update x1 based on new width
                y0, y1 = current_coords[1], current_coords[3]
                new_x1 = start_x + px_width
                
                # Ensure visible minimum width
                if px_width < 2: new_x1 = start_x + 2
                
                # Update Bar Geometry
                sensor_canvas.coords(tag_name, start_x, y0, new_x1, y1)
                
                # Color Logic
                if ratio > 0.8:
                    sensor_canvas.itemconfig(tag_name, fill="#dc3545") # Red warning
                else:
                    sensor_canvas.itemconfig(tag_name, fill=bar_colors[i]) # Normal Color
                
                # === UPDATE TEXT VALUE POSITION ===
                # Place text 10 pixels to the right of the bar end
                text_x = new_x1 + 10 
                # Vertical center of the bar
                mid_y = (y0 + y1) / 2
                
                sensor_canvas.coords(val_tag_name, text_x, mid_y)
                
                # === FORMAT TEXT (2 decimals + PPM logic) ===
                if i < 3: # MQ4, MQ135, TGS
                    display_text = f"{val:.2f} PPM"
                else:     # NIR / Spectral
                    display_text = f"{val:.2f}"
                
                sensor_canvas.itemconfig(val_tag_name, text=display_text)

        except Exception as e:
            print(f"Graph update error: {e}")

def update_ui(result):
    """
    Updates all UI elements based on the final result dictionary.
    """
    # === Initialize variables to safe defaults ===
    item_name = "Unknown"
    pred_name = "Unknown"
    info = PRODUCE_INFO.get("banana") 

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

        # === 2. Get the produce info ===
        info = PRODUCE_INFO.get(item_name, PRODUCE_INFO.get("banana")) 
        
        # === 3. Update Image ===
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
        
        # === 4. Update Text Labels ===
        produce_name_label.config(text=info["name"])
        
        # === 5. Update SENSORS ===
        update_sensor_graph_ui(sensor_values)
        
        # === Dynamically set label based on produce type ===
        produce_type = info.get("type", "fruit")
        if produce_type == "vegetable":
            label_text = "Freshness Level:"
            ripeness_label.place(x=10, y=335)
            ripeness_label.config(text=f"{label_text} {pred_name}", font=("Poppins", 13, "bold"))
        else:
            label_text = "Ripeness Level:"
            ripeness_label.config(text=f"{label_text} {pred_name}")
        
        draw_gauge(gauge_canvas, pred_name)

        # === 6. Update Shelf Life ===
        if pred_name in ("Rotten", "Overripe"):
            shelf_life_text.config(text="Item is past its prime.")
        elif days_to_rotten is not None and rotten_date_str is not None:
            shelf_life_text.config(text=f"Approx. {days_to_rotten} days remaining.\n(Est. rotten date: {rotten_date_str})")
        else:
            shelf_life_text.config(text="...")

        # === 7. Update Nutrients ===
        nutrient_text.config(text="Getting nutrients...")
        threading.Thread(target=fetch_and_update_nutrients, args=(info["name"],pred_name), daemon=True).start()

    except Exception as e:
        print(f"❌ Error in update_ui: {e}")
        try:
            ripeness_label.config(text="Ripeness Level: Error")
        except Exception:
            pass 


def run_full_analysis_pipeline():
    """
    Main function that orchestrates the entire process.
    """
    # --- 1. Clear UI for new scan ---
    ripeness_label.config(text="        Identifying..")
    ripeness_label.place(x=75, y=335)
    ripeness_label.config(font=("Poppins", 14, "bold"))
    produce_name_label.config(text="...")
    nutrient_text.config(text="...")
    shelf_life_text.config(text="...") 
    
    # === Reset Graph ===
    update_sensor_graph_ui([0, 0, 0, 0]) 
    
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
        # Map item names to functions
        if item_name == "banana": analysis_result = run_banana_analysis()
        elif item_name == "carrot": analysis_result = run_carrot_analysis()
        elif item_name == "mango": analysis_result = run_mango_analysis()
        elif item_name == "strawberry": analysis_result = run_strawberry_analysis()
        elif item_name == "apple": analysis_result = run_apple_analysis()
        elif item_name == "tomato": analysis_result = run_tomato_analysis()
        elif item_name == "bellpepper": analysis_result = run_bellpepper_analysis()
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
    """Runs calibration and reloads settings."""
    # Update UI to show we are working
    scan_button.config(text="Calibrating...", bg="#ffc107", state="disabled")
    ripeness_label.config(text="Calibrating Sensors...")
    root.update()

    # Run the calibration logic
    success = calibrate_now.perform_calibration()

    if success:
        # RELOAD the config so the code sees the new numbers immediately
        importlib.reload(sensor_config)
        print("🔄 Configuration Reloaded!")

        ripeness_label.config(text="Calibration Done!")
        scan_button.config(text="Scan Produce", bg="#2b8a3e", state="normal")

        # Optional: Show the new "zeroed" values on screen immediately
        # You could trigger a quick sensor read here if you wanted
    else:
        ripeness_label.config(text="Calibration Failed")
        scan_button.config(text="Scan Produce", bg="#2b8a3e", state="normal")

# ==============================
# === GreenMate Tkinter UI ====
# ==============================

# --- Tkinter Setup ---
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

# --- Load Logo ---
logo_path = "/home/device/ML_model/logo.png"
try:
    logo_img = Image.open(logo_path).resize((120, 60))
    logo_photo = ImageTk.PhotoImage(logo_img)
except Exception as e:
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
tk.Label(header_frame, text="Eat Fresh. Live Healthy.", bg="#D3F0D8", font=("Poppins", 10, "italic")).pack()

# ==================================
# === LEFT SIDE (Image & Gauge) ====
# ==================================

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
    """Draws the ripeness gauge."""
    canvas.delete("all")
    cx, cy, r = 100, 100, 80
    
    # Background arcs
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
# === RIGHT SIDE (Sensors Graph - HORIZONTAL) ===
# ====================================================
sensor_frame = tk.Frame(content_frame, bg="#D3F0D8")
sensor_frame.place(x=380, y=0)
tk.Label(sensor_frame, text="Quality Metrics:", font=("Poppins", 12, "bold"), bg="#D3F0D8").pack(anchor="w", pady=(0, 5))

# --- GRAPH SETUP ---
graph_width = 415
graph_height = 160 
bar_height = 20
spacing = 30       # Vertical space between bars
start_y = 10      # Top margin
start_x = 105      # Left margin (space for labels "Methane", etc.)

sensor_canvas = tk.Canvas(sensor_frame, width=graph_width, height=graph_height, bg="#D3F0D8", highlightthickness=0)
sensor_canvas.pack()

# Draw Vertical Axis Line
sensor_canvas.create_line(start_x, 10, start_x, graph_height-10, fill="#7f8c8d", width=2)

# --- CONFIGURATION ---
internal_tags = ["MQ4", "MQ135", "TGS", "NIR"]
display_labels = ["Methane", "Ammonia/CO2", "VOCs", "Spectral Data"]
bar_colors = ["#17a2b8", "#28a745", "#ffc107", "#6f42c1"]

for i in range(4):
    y0 = start_y + (i * spacing)
    y1 = y0 + bar_height
    
    # Initial Width (0)
    x0 = start_x
    x1 = start_x + 1 # Start with tiny bar
    
    tag_name = internal_tags[i]
    label_text = display_labels[i]
    
    # 1. Draw Text Label on LEFT
    sensor_canvas.create_text(start_x - 10, (y0 + y1)/2, text=label_text, 
                              font=("Poppins", 9, "bold"), fill="#333333", anchor="e")
    
    # 2. Draw Bar
    sensor_canvas.create_rectangle(x0, y0, x1, y1, fill=bar_colors[i], outline="", tags=tag_name)
    
    # 3. Create Value Text Object (Initially '0.00' on right side)
    val_tag_name = tag_name + "_val"
    sensor_canvas.create_text(x1 + 10, (y0 + y1)/2, text="0.00", 
                              font=("Poppins", 8, "bold"), fill="#333333", anchor="w", tags=val_tag_name)

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
right_panel_canvas.place(x=380, y=155, width=panel_width, height=panel_height)
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
    scan_icon_img = Image.open(icon_path).resize((30, 30))
    scan_icon_photo = ImageTk.PhotoImage(scan_icon_img)
    btn_compound = "left"
    btn_image = scan_icon_photo
    btn_padx = 10
except Exception as e:
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
                        command=run_analysis_thread)
scan_button.pack(pady=0)
if scan_icon_photo:
    scan_button.image = scan_icon_photo

# ==============================
# === GPIO Check Loop =====
# ==============================

def check_button():
    try:
        # Check if button is pressed
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            start_time = time.time()
            long_press_triggered = False  # Flag to track if we did a long press

            # Wait while button is HELD down
            while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                time.sleep(0.1)
                
                # If held for more than 3 seconds...
                if not long_press_triggered and (time.time() - start_time > 3):
                    print("🔘 Long Press Detected: Starting Calibration...")
                    run_auto_calibration()
                    long_press_triggered = True 
                    # We do NOT return here yet. We wait for the user 
                    # to release the button inside this while loop.

            # === Button has now been released ===
            
            # If it was a SHORT press (and we didn't trigger calibration)
            if not long_press_triggered:
                duration = time.time() - start_time
                if scan_button["state"] == "normal":
                    print("🔘 Short Press Detected: Scanning...")
                    run_analysis_thread()

        # CRITICAL: Always schedule the next check!
        root.after(100, check_button)

    except tk.TclError:
        pass # Window was closed

print("\n🔹 System ready. Launching GreenMate UI...\n")
root.after(300, check_button)
root.mainloop()

# Clean up GPIO on exit
print("🛑 Exiting... cleaning up GPIO.")
GPIO.cleanup()