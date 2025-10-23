import time
import board
import busio
import torch
import torch.nn as nn
import joblib
import numpy as np
import RPi.GPIO as GPIO
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import math
import subprocess
import adafruit_as726x
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
import cv2
import os


# === Load YOLO Model ===
model = YOLO("/home/device/ML_model/best_my_model2.pt")

# === Setup I2C for ADS1015 & AS7263 ===
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1015(i2c)

# === Create analog input channels ===
mq4_chan = AnalogIn(ads, 0)
mq135_chan = AnalogIn(ads, 1)
tgs2602_chan = AnalogIn(ads, 2)

# === Model Setup ===
DEVICE = torch.device("cpu")
NUM_CLASSES = 3


# === Feature Extractors ===
class EfficientFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.base = (
            EfficientNet.from_pretrained('efficientnet-b3')
            if pretrained
            else EfficientNet.from_name('efficientnet-b3')
        )

    def forward(self, x):
        features = self.base.extract_features(x)
        out = nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)
        return out


class SensorNetFeat(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class EarlyFusionModel(nn.Module):
    def __init__(self, cnn_feat_extractor, sensor_feat_extractor, img_dim, sensor_dim, num_classes=3):
        super().__init__()
        self.cnn = cnn_feat_extractor
        self.sensor = sensor_feat_extractor
        hidden = 256
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + sensor_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, sensor):
        img_feat = self.cnn(img)
        sensor_feat = self.sensor(sensor)
        img_feat = img_feat * 3.0
        x = torch.cat([img_feat, sensor_feat], dim=1)
        out = self.fusion(x)
        return out


# === Load Model and Preprocessors ===
def load_model():
    cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
    cnn_extractor.base.load_state_dict(
        torch.load("Early_Fusion_Banana/Banana_CNN.pth", map_location=DEVICE),
        strict=False
    )
    cnn_extractor.eval()

    sensor_feat = SensorNetFeat().to(DEVICE)
    sensor_feat.load_state_dict(
        torch.load("Early_Fusion_Banana/banana_900_sensor_modelOct.pth", map_location=DEVICE),
        strict=False
    )
    sensor_feat.eval()

    img_dim = cnn_extractor(torch.randn(1, 3, 224, 224)).shape[1]
    sensor_dim = sensor_feat(torch.randn(1, 4)).shape[1]

    fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, NUM_CLASSES).to(DEVICE)
    fusion_model.load_state_dict(
        torch.load("Early_Fusion_Banana/banana_early_fusion_model_October.pth", map_location=DEVICE)
    )
    fusion_model.eval()
    return fusion_model


def load_scaler():
    return joblib.load('Early_Fusion_Banana/sensor_scaler.save')


def load_label_encoder():
    return joblib.load('Early_Fusion_Banana/label_encoder_October.save')


fusion_model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()


# === Image Transform ===
val_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# === Capture Image (USB Webcam) ===
def capture_image(filename="captured_image.jpg"):
    try:
        as7263 = adafruit_as726x.AS726x_I2C(i2c)

        # Turn ON illumination LED before capture
        as7263.driver_led = True
        print("💡 AS7263 LED turned ON for image capture.")
        time.sleep(0.5)

        subprocess.run([
            "fswebcam",
            "-d", "/dev/video0",
            "-r", "1280x720",
            "--no-banner",
            filename
        ], check=True)

        print(f"✅ Image captured successfully: {filename}")
        as7263.driver_led = False
        return Image.open(filename)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error capturing image: {e}")
        as7263.driver_led = False
        return None


# === Read Sensors (with ppm conversion) ===
def read_sensors(sample_id="banana1"):
    try:
        # MQ4
        mq4_v = mq4_chan.voltage
        rs = (5 - mq4_v) * 1000 / mq4_v
        ratio = rs / 10000
        ppm_mq4 = 10 ** (-0.38 * math.log10(ratio) + 1.58)
        print(f"[{sample_id}] MQ4: {round(ppm_mq4, 2)} ppm")

        # MQ135
        mq135_v = mq135_chan.voltage-0.044
        rs = (5 - mq135_v) * 1000 / mq135_v
        ratio = rs / 10000
        ppm_mq135 = 10 ** (-0.38 * math.log10(ratio) + 1.58)
        print(f"[{sample_id}] MQ135: {round(ppm_mq135, 2)} ppm")

        # TGS2602
        tgs2602_v = tgs2602_chan.voltage
        rs = (5 - tgs2602_v) * 1000 / tgs2602_v
        ratio = rs / 10000
        ppm_tgs2602 = 10 ** (-0.38 * math.log10(ratio) + 1.58)
        print(f"[{sample_id}] TGS2602: {round(ppm_tgs2602, 2)} ppm")

        # AS7263 (NIR)
        as7263 = adafruit_as726x.AS726x_I2C(i2c)
        as7263.driver_led = True
        time.sleep(0.5)

        while not as7263.data_ready:
            time.sleep(0.05)

        violet = as7263.violet
        as7263.driver_led = False
        print(f"[{sample_id}] NIR (violet): {violet}")

        return [ppm_mq4, ppm_mq135, ppm_tgs2602, violet]

    except Exception as e:
        print(f"[{sample_id}] Sensor Error: {e}")
        return [0, 0, 0, 0]


# === Prediction Functions ===
def predict_ripeness(image, sensor_values):
    img = image.convert('RGB')
    img = val_img_transform(img).unsqueeze(0).to(DEVICE)
    sensor_np = np.array(sensor_values).reshape(1, -1).astype(np.float32)
    sensor_scaled = scaler.transform(sensor_np)
    sensor_tensor = torch.tensor(sensor_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = fusion_model(img, sensor_tensor)
        pred_class = outputs.argmax(dim=1).item()

    return pred_class


def load_regression_model():
    return joblib.load("Early_Fusion_Banana/ripeness_regression_model_new2.pkl")


regression_model = load_regression_model()


def predict_days_to_rotten(sensor_values, predicted_label):
    new_sample = pd.DataFrame([{
        "ppm1(mq4)": sensor_values[0],
        "ppm2(mq135)": sensor_values[1],
        "ppm3(tgs2602)": sensor_values[2],
        "NIR values": sensor_values[3],
        "label": predicted_label
    }])
    days_pred = regression_model.predict(new_sample)
    return int(round(days_pred[0]))


# === GPIO Button Setup ===
BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("\n🔹 System ready. Launching GreenMate UI...\n")
time.sleep(1)

# ==============================
# === GreenMate Tkinter UI ====
# ==============================
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import threading
import math # Needed for the gauge

# --- Tkinter Setup ---
root = tk.Tk()
root.title("GreenMate – Freshness Ripeness Simplified")
root.geometry("800x480")
root.configure(bg="#D3F0D8")
root.resizable(False, False)

# --- Load Logo ---
logo_path = "/home/device/ML_model/logo.png"
try:
    logo_img = Image.open(logo_path).resize((120, 60)) # Adjusted size for header
    logo_photo = ImageTk.PhotoImage(logo_img)
except Exception as e:
    print(f"Logo load error: {e}")
    logo_photo = None

# --- Layout Frames ---
top_frame = tk.Frame(root, bg="#D3F0D8")
top_frame.pack(fill="x", pady=(10, 5))
content_frame = tk.Frame(root, bg="#D3F0D8")
content_frame.pack(expand=True, fill="both")
# No bottom_frame, button will be in the right panel


# ==============================================================
# === Core Logic Functions (MOVED UP to fix NameError) =====
# ==============================================================

# --- Function to Update UI ---
def update_ui(pred_name, sensor_values, img_path):
    try:
        # Load and resize the image for the NEW canvas size
        img = Image.open(img_path)
        img_w, img_h = img.size
        # Maintain aspect ratio to fit 268x135 box (2px padding)
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

        # Place image on the label inside the canvas
        image_label.configure(image=new_photo, bg="white")
        image_label.image = new_photo
        # Re-center the label in the canvas
        image_canvas.delete("image_window") # Remove old window
        image_canvas.create_window(135, 68, window=image_label, anchor="center", tags="image_window")
        
    except Exception as e:
        print(f"UI Image update error: {e}")

    # Update sensor labels
    sensor_labels["MQ 4 Sensor"].config(text=f"{sensor_values[0]:.2f} ppm")
    sensor_labels["MQ 135 Sensor"].config(text=f"{sensor_values[1]:.2f} ppm")
    sensor_labels["TGS2602 Sensor"].config(text=f"{sensor_values[2]:.2f} ppm")
    sensor_labels["NIR Spectrometer"].config(text=f"{sensor_values[3]:.2f}")
    
    # Update ripeness label and gauge
    ripeness_label.config(text=f"Ripeness Level: {pred_name}")
    draw_gauge(gauge_canvas, pred_name) # <-- This updates the gauge

    # Update nutrients and suggestions
    nutrient_text.config(text="Vitamin B6, Vitamin C, Fiber, Potassium")
    
    if pred_name == "Ripe":
        suggestion_text.config(text="Best for direct eating or smoothies.\nAvoid refrigeration for natural ripening.")
    elif pred_name == "Underripe":
        suggestion_text.config(text="Store at room temperature. Will be ripe in a few days. Good for cooking.")
    elif pred_name == "Overripe":
        suggestion_text.config(text="Ideal for baking (banana bread) or freezing for later use in smoothies.")
    else:
        suggestion_text.config(text="Please scan again.")


# --- Main Analysis Function ---
def run_analysis():
    # --- Clear UI for new scan ---
    ripeness_label.config(text="Ripeness Level: Scanning...")
    nutrient_text.config(text="...")
    suggestion_text.config(text="...")
    sensor_labels["MQ 4 Sensor"].config(text="... ppm")
    sensor_labels["MQ 135 Sensor"].config(text="... ppm")
    sensor_labels["TGS2602 Sensor"].config(text="... ppm")
    sensor_labels["NIR Spectrometer"].config(text="...")
    draw_gauge(gauge_canvas, "...")
    
    # Clear image
    default_photo = ImageTk.PhotoImage(Image.new("RGB", (1, 1), color="#ffffff"))
    image_label.configure(image=default_photo, bg="white")
    image_label.image = default_photo

    root.update_idletasks() # Force UI update
    # --- End Clear UI ---

    filename = f"/home/device/ML_model/image_{int(time.time())}.jpg"
    image = capture_image(filename=filename)
    if image is None:
        print("⚠️ Capture failed.")
        ripeness_label.config(text="Ripeness Level: Capture Failed")
        return

    img_cv = cv2.imread(filename)
    results = model(filename)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = map(int, largest[:4])
        cropped = img_cv[y1:y2, x1:x2]
        os.makedirs("cropped_output", exist_ok=True)
        save_path = os.path.join("cropped_output", "cropped_object.jpg")
        cv2.imwrite(save_path, cropped)
        cropped_image = Image.open(save_path)
    else:
        print("⚠️ No object detected, using full image.")
        cropped_image = image
        save_path = filename

    sensor_values = read_sensors("banana1")
    pred_idx = predict_ripeness(cropped_image, sensor_values)
    pred_name = label_encoder.classes_[pred_idx]

    # Update the UI with all results
    update_ui(pred_name, sensor_values, save_path)


# --- Run in Thread (so UI doesn’t freeze) ---
def run_analysis_thread():
    # Disable button
    scan_button.config(state="disabled", text="Scanning...")
    threading.Thread(target=run_analysis_with_button_reset, daemon=True).start()

def run_analysis_with_button_reset():
    """Wraps analysis to re-enable button after completion."""
    try:
        run_analysis()
    except Exception as e:
        print(f"Error in analysis thread: {e}")
        ripeness_label.config(text="Ripeness Level: Error")
    finally:
        # Re-enable button
        scan_button.config(state="normal", text="Scan Produce")


# --- Header ---
header_frame = tk.Frame(top_frame, bg="#D3F0D8")
header_frame.pack(side="top", pady=5)

if logo_photo:
    tk.Label(header_frame, image=logo_photo, bg="#D3F0D8").pack(pady=(0,2)) # Logo on top
tk.Label(header_frame, text="Freshness Ripeness Simplified", bg="#D3F0D8", font=("Poppins", 9, "normal")).pack() # Subtitle



# ==================================
# === LEFT SIDE (Image & Gauge) ====
# ==================================

# --- Rounded Corner Function (for Image) ---
def create_rounded_rect(canvas, x, y, w, h, r, **kwargs):
    """Draws a rounded rectangle on a canvas."""
    canvas.create_arc(x, y, x+2*r, y+2*r, start=90, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x+w-2*r, y, x+w, y+2*r, start=0, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x, y+h-2*r, x+2*r, y+h, start=180, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_arc(x+w-2*r, y+h-2*r, x+w, y+h, start=270, extent=90, style=tk.PIESLICE, **kwargs)
    canvas.create_rectangle(x+r, y, x+w-r, y+h, **kwargs)
    canvas.create_rectangle(x, y+r, x+w, y+h-r, **kwargs)

# --- Produce Image Canvas (NEW SIZE) ---
image_canvas_width = 270
image_canvas_height = 137
image_canvas = tk.Canvas(content_frame, width=image_canvas_width, height=image_canvas_height, bg="#D3F0D8", highlightthickness=0)
image_canvas.place(x=50, y=20)
# Draw the white rounded background
create_rounded_rect(image_canvas, 0, 0, image_canvas_width, image_canvas_height, 15, fill="white", outline="")

# Create a label for the image (it will be placed inside the canvas)
default_img = Image.new("RGB", (1, 1), color="#ffffff") # Start with blank
default_photo = ImageTk.PhotoImage(default_img)
image_label = tk.Label(image_canvas, image=default_photo, bg="white")
# Center window in new canvas size
image_canvas.create_window(image_canvas_width/2, image_canvas_height/2, window=image_label, anchor="center", tags="image_window")


# --- Produce Name Label (CENTERED UNDER IMAGE) ---
produce_name_label = tk.Label(content_frame, text="Banana", font=("Poppins", 16, "bold"), bg="#D3F0D8")
# New x is canvas start (50) + half its width (270/2=135) = 185
# New y is canvas start (20) + its height (137) + padding (5) = 162
produce_name_label.place(x=185, y=162, anchor="n") 


# --- Ripeness Gauge ---
gauge_canvas = tk.Canvas(content_frame, width=200, height=120, bg="#D3F0D8", highlightthickness=0)
gauge_canvas.place(x=85, y=210) # Centered under new image position

def draw_gauge(canvas, ripeness_name):
    """Draws the ripeness gauge on the canvas."""
    canvas.delete("all")
    cx, cy, r = 100, 100, 80 # Center x, y, radius
    
    # Draw background arcs
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=-60, 
                      fill="#28a745", outline="#28a745", style=tk.ARC, width=20) # Green
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=120, extent=-60, 
                      fill="#ffc107", outline="#ffc107", style=tk.ARC, width=20) # Yellow
    canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=60, extent=-60, 
                      fill="#dc3545", outline="#dc3545", style=tk.ARC, width=20) # Red

    # Map ripeness name to angle
    if ripeness_name == "Unripe":
        angle_deg = 30 # Pointing between yellow/red
    elif ripeness_name == "Ripe":
        angle_deg = 70 # Pointing at green
    elif ripeness_name == "Rotten":
        angle_deg = 150 # Pointing at red
    else: # Default/unknown state
        angle_deg = 90 # Pointing straight up (yellow)
        
    angle_rad = math.radians(angle_deg)
    lx = cx - (r * 0.85) * math.cos(angle_rad) # Needle end X
    ly = cy - (r * 0.85) * math.sin(angle_rad) # Needle end Y
    
    # Draw needle
    canvas.create_line(cx, cy, lx, ly, fill="black", width=4)
    # Draw pivot
    canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill="black", outline="white", width=2)

# --- Initial Gauge Draw ---
draw_gauge(gauge_canvas, "—") 

# --- Ripeness Label (NOW BOLD) ---
ripeness_label = tk.Label(content_frame, text="Ripeness Level: —", font=("Poppins", 14, "bold"), bg="#D3F0D8")
ripeness_label.place(x=105, y=335) # Centered under gauge


# =====================================
# === RIGHT SIDE (Sensors & Panel) ===
# =====================================

# --- Sensor Display (NOW on main content_frame) ---
sensor_frame = tk.Frame(content_frame, bg="#D3F0D8")
sensor_frame.place(x=380, y=20)

tk.Label(sensor_frame, text="Sensors in active:", font=("Poppins", 12, "bold"), bg="#D3F0D8").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

sensor_labels = {}
for i, name in enumerate(["MQ 4 Sensor", "MQ 135 Sensor", "TGS2602 Sensor", "NIR Spectrometer"], start=1):
    # Labels are normal font
    tk.Label(sensor_frame, text=f"{name}:", font=("Poppins", 12, "normal"), bg="#D3F0D8").grid(row=i, column=0, sticky="w", padx=(0, 20))
    # Values are bold and right-aligned
    lbl = tk.Label(sensor_frame, text="—", font=("Poppins", 12, "bold"), bg="#D3F0D8", width=10, anchor="e")
    lbl.grid(row=i, column=1, padx=10, sticky="e")
    sensor_labels[name] = lbl


# --- Rounded Corner Panel Function (for Right Panel) ---
def create_rounded_panel(canvas, x, y, w, h, r):
    """Draws a white panel with rounded top corners on the canvas."""
    border_color = "#bbbbbb"
    fill_color = "#ffffff"
    
    # Draw the fill
    canvas.create_rectangle(x, y+r, x+w, y+h, fill=fill_color, outline="")
    canvas.create_rectangle(x+r, y, x+w-r, y+h, fill=fill_color, outline="")
    canvas.create_arc(x, y, x+2*r, y+2*r, start=90, extent=90, fill=fill_color, outline="", style=tk.PIESLICE)
    canvas.create_arc(x+w-2*r, y, x+w, y+2*r, start=0, extent=90, fill=fill_color, outline="", style=tk.PIESLICE)
    
    # Draw the border
    canvas.create_arc(x, y, x+2*r, y+2*r, start=90, extent=90, style=tk.ARC, outline=border_color, width=1)
    canvas.create_arc(x+w-2*r, y, x+w-1, y+2*r, start=0, extent=90, style=tk.ARC, outline=border_color, width=1)
    canvas.create_line(x+r, y, x+w-r, y, fill=border_color, width=1)
    canvas.create_line(x, y+r, x, y+h-1, fill=border_color, width=1)
    canvas.create_line(x+w-1, y+r, x+w-1, y+h-1, fill=border_color, width=1)
    canvas.create_line(x, y+h-1, x+w-1, y+h-1, fill=border_color, width=1)


# --- White Panel Canvas (NOW shorter and lower) ---
panel_width = 400
panel_height = 230 # Made shorter
right_panel_canvas = tk.Canvas(content_frame, bg="#D3F0D8", bd=0, highlightthickness=0)
right_panel_canvas.place(x=380, y=170, width=panel_width, height=panel_height) # Moved down

# Draw the rounded shape onto the canvas
create_rounded_panel(right_panel_canvas, 0, 0, panel_width, panel_height, 20) # 20px radius

# --- Inner Frame (to hold widgets) ---
# This frame is placed *inside* the canvas
inner_panel = tk.Frame(right_panel_canvas, bg="#ffffff")
right_panel_canvas.create_window(1, 1, window=inner_panel, anchor="nw", 
                                 width=panel_width-2, height=panel_height-2)

# --- Nutrients & Suggestions (parent is inner_panel) ---
nutrient_label = tk.Label(inner_panel, text="Nutritients:", font=("Poppins", 12, "bold"), bg="#ffffff")
nutrient_label.pack(side="top", anchor="w", padx=20, pady=(15,0))
nutrient_text = tk.Label(inner_panel, text="", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
nutrient_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")

suggestion_label = tk.Label(inner_panel, text="Suggestions:", font=("Poppins", 12, "bold"), bg="#ffffff")
suggestion_label.pack(side="top", anchor="w", padx=20, pady=(15,0))
suggestion_text = tk.Label(inner_panel, text="", font=("Poppins", 11, "normal"), bg="#ffffff", justify="left", wraplength=360, anchor="w")
suggestion_text.pack(side="top", anchor="w", padx=20, pady=2, fill="x")


# --- Load Button Icon ---
try:
    icon_path = "/home/device/ML_model/scan_icon.png" # <--- CHECK THIS PATH
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

# --- UI Button (parent is inner_panel) (CENTERED AND MOVED UP) ---
button_frame = tk.Frame(inner_panel, bg="#ffffff")
button_frame.pack(side="bottom", fill="x", pady=15) # Use a frame to center easily

scan_button = tk.Button(button_frame, text="Scan Produce", font=("Poppins", 13, "bold"),
                        bg="#2b8a3e", fg="white", relief="raised", 
                        image=btn_image, compound=btn_compound,
                        padx=btn_padx,
                        command=run_analysis_thread)
scan_button.pack(pady=5) # pack in center of the frame
if scan_icon_photo:
    scan_button.image = scan_icon_photo # Keep reference


# ==============================
# === GPIO Check Loop =====
# ==============================

def check_button():
    if GPIO.input(BUTTON_PIN) == GPIO.LOW:
        # Check if button is already disabled (scan in progress)
        if scan_button["state"] == "normal":
            print("🔘 Physical button pressed!")
            run_analysis_thread()
    root.after(300, check_button) # Check every 300ms

root.after(300, check_button)
root.mainloop()

# Clean up GPIO on exit
GPIO.cleanup()