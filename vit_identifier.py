from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 🔹 Load the model and preprocessor from Hugging Face
model_id = "shahmi0519/fypvit"

processor = ViTImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id)

# 🔹 Function to predict fruit or vegetable type
def predict_item(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_idx]
    #print(f"🧠 Detected item: {label}")
    return label

if __name__ == "__main__":
    test_image = "captured_item.jpg"  # your captured image path
    predict_item(test_image)
