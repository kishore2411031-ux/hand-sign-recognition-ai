import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model/hand_sign_model.pth"

# 🔥 Manually define your class names (IMPORTANT)
CLASS_NAMES = [
    "Like",
    "Dislike",
    "Stop",
    "Fist",
    "Peace",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "A",
    "B",
    "C",
    "D",
    "E",
    "Hello",
    "Yes",
    "No"
]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- STREAMLIT UI ----------------
st.title("🤖 AI Hand Sign Recognition")
st.write("Upload an image to detect hand sign")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    prediction = CLASS_NAMES[predicted.item()]
    confidence = confidence.item() * 100

    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence:.2f}%")
