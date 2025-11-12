import os
import time
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image
import torch
import torch.nn as nn
import joblib
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import math
import subprocess
import adafruit_as726x
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
from ultralytics import YOLO
import board, busio
import RPi.GPIO as GPIO

# === Load YOLO Model ===
model = YOLO("/home/device/ML_model/best_my_model2.pt")

# === Setup I2C for ADS1015 & AS7263 ===
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1015(i2c)

# === NEW: Setup GPIO for Flasher Light ===
FLASHER_PIN = 16  # This is GPIO16 (BCM naming)
GPIO.setmode(GPIO.BCM)
GPIO.setup(FLASHER_PIN, GPIO.OUT)
GPIO.output(FLASHER_PIN, GPIO.LOW) # Ensure it's off to start

# === Create analog input channels ===
mq4_chan = AnalogIn(ads, 0)
mq135_chan = AnalogIn(ads, 1)
tgs2602_chan = AnalogIn(ads, 2)

# === Device & Classes ===
DEVICE = torch.device("cpu")
NUM_CLASSES = 3

# === Feature Extractors ===
class EfficientFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b3') if pretrained else EfficientNet.from_name('efficientnet-b3')

    def forward(self, x):
        features = self.base.extract_features(x)
        return nn.functional.adaptive_avg_pool2d(features, 1).reshape(features.shape[0], -1)

class SensorNetFeat(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim), nn.ReLU()
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
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, sensor):
        img_feat = self.cnn(img) * 3.0
        sensor_feat = self.sensor(sensor)
        return self.fusion(torch.cat([img_feat, sensor_feat], dim=1))

# === Load models & preprocessors ===
def load_model():
    cnn_extractor = EfficientFeatureExtractor(pretrained=False).to(DEVICE)
    cnn_extractor.base.load_state_dict(torch.load("Early_Fusion_Apple/best_apple_model.pth", map_location=DEVICE), strict=False)
    cnn_extractor.eval()
    sensor_feat = SensorNetFeat().to(DEVICE)
    sensor_feat.load_state_dict(torch.load("Early_Fusion_Apple/apple_900_sensor_model.pth", map_location=DEVICE), strict=False)
    sensor_feat.eval()
    img_dim = cnn_extractor(torch.randn(1, 3, 224, 224)).shape[1]
    sensor_dim = sensor_feat(torch.randn(1, 4)).shape[1]
    fusion_model = EarlyFusionModel(cnn_extractor, sensor_feat, img_dim, sensor_dim, NUM_CLASSES).to(DEVICE)
    fusion_model.load_state_dict(torch.load("Early_Fusion_Apple/apple_early_fusion_model_Nov.pth", map_location=DEVICE))
    fusion_model.eval()
    return fusion_model

def load_scaler():
    return joblib.load('Early_Fusion_Apple/apple_sensor_scaler.save')

def load_label_encoder():
    return joblib.load('Early_Fusion_Apple/apple_label_encoder.save')

def load_regression_model():
    return joblib.load("Early_Fusion_Apple/apple_regression_model.pkl")

fusion_model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()
regression_model = load_regression_model()

# === Image Transform ===
val_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# === Capture Image ===
def capture_image(filename="captured_image.jpg"):
    try:
        as7263 = adafruit_as726x.AS726x_I2C(i2c)
        as7263.driver_led = True
        GPIO.output(FLASHER_PIN, GPIO.HIGH)
        time.sleep(0.5)
        subprocess.run(["fswebcam","-d","/dev/video0","-r","1280x720","--no-banner", filename], check=True)
        as7263.driver_led = False
        GPIO.output(FLASHER_PIN, GPIO.LOW)
        return Image.open(filename)
    except subprocess.CalledProcessError as e:
        as7263.driver_led = False
        print(f"❌ Error capturing image: {e}")
        return None

# === Read Sensors ===
def read_sensors(sample_id="apple1"):
    try:
        # MQ4
        mq4_v = mq4_chan.voltage-0.010
        rs = (5 - mq4_v) * 1000 / mq4_v
        ratio = rs / 10000
        ppm_mq4 = 10 ** (-0.38 * math.log10(ratio) + 1.58)

        # MQ135
        mq135_v = mq135_chan.voltage-0.058
        rs = (5 - mq135_v) * 1000 / mq135_v
        ratio = rs / 10000
        ppm_mq135 = 10 ** (-0.38 * math.log10(ratio) + 1.58)

        # TGS2602
        tgs2602_v = tgs2602_chan.voltage
        rs = (5 - tgs2602_v) * 1000 / tgs2602_v
        ratio = rs / 10000
        ppm_tgs2602 = 10 ** (-0.38 * math.log10(ratio) + 1.58)

        # AS7263 (NIR)
        as7263 = adafruit_as726x.AS726x_I2C(i2c)
        as7263.driver_led = True
        time.sleep(0.5)
        while not as7263.data_ready:
            time.sleep(0.05)
        violet = as7263.violet+30.35116577148437
        as7263.driver_led = False

        return [ppm_mq4, ppm_mq135, ppm_tgs2602, violet]
    except Exception as e:
        print(f"[{sample_id}] Sensor Error: {e}")
        return [0, 0, 0, 0]

# === Prediction ===
def predict_ripeness(image, sensor_values):
    img = val_img_transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    sensor_tensor = torch.tensor(scaler.transform(np.array(sensor_values).reshape(1,-1)), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_idx = fusion_model(img, sensor_tensor).argmax(dim=1).item()
    return pred_idx

def predict_days_to_rotten(sensor_values, pred_label):
    df = pd.DataFrame([{
        "ppm1(mq4)": sensor_values[0],
        "ppm2(mq135)": sensor_values[1],
        "ppm3(tgs)": sensor_values[2],
        "nir value": sensor_values[3],
        "label": pred_label
    }])
    return int(round(regression_model.predict(df)[0]))

# === Main Single-Run Function ===
def run_analysis():
    filename = f"/home/device/ML_model/image_{int(time.time())}.jpg"
    image = capture_image(filename=filename)
    if image is None: return None

    img_cv = cv2.imread(filename)
    results = model(filename)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return None

    largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1,y1,x2,y2 = map(int, largest[:4])
    cropped = img_cv[y1:y2,x1:x2]

    os.makedirs("cropped_output", exist_ok=True)
    crop_path = os.path.join("cropped_output","cropped_object.jpg")
    cv2.imwrite(crop_path, cropped)
    cropped_image = Image.open(crop_path)

    sensor_values = read_sensors("apple1")
    pred_idx = predict_ripeness(cropped_image, sensor_values)
    pred_name = label_encoder.classes_[pred_idx]
    days_to_rotten = predict_days_to_rotten(sensor_values, pred_name)
    rotten_date_str = (datetime.today() + timedelta(days=days_to_rotten)).strftime("%Y-%m-%d")

    return {
        "item":"apple",
        "ripeness": pred_name,
        "days_to_rotten": days_to_rotten,
        "estimated_rotten_date": rotten_date_str,
        "sensor_values": sensor_values,
        "image_path": filename
    }

# === Run Analysis ===
if __name__=="__main__":
    result = run_analysis()
    if result:
        print(f"📸 Image: {result['image_path']}")
        print(f"🍓 Item: {result['item']}")
        print(f"🍌 Predicted Ripeness: {result['ripeness']}")
        print(f"🕒 Days to Rotten: {result['days_to_rotten']}")
        print(f"📅 Estimated Rotten Date: {result['estimated_rotten_date']}")
        print(f"🔬 Sensor Values: {result['sensor_values']}\n")
