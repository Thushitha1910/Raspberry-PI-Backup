import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("✅ Gemini API key loaded successfully!")

# list models to confirm
for model in genai.list_models():
    print(model.name, "|", model.supported_generation_methods)