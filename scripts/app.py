import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "model/hand_sign_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 18

# ⚠️ CHANGE THIS ORDER UNTIL 👍 SHOWS "Like"
CLASS_NAMES = [
    "Three",   # 0
    "Like",    # 1
    "One",
    "Two",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Stop",
    "Rock",
    "Call",
    "Peace",
    "Okay",
    "ThumbsDown",
    "Fist"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

st.set_page_config(page_title="AI Hand Sign Recognition", layout="centered")
st.title("🤖 AI Hand Sign Recognition")

option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

def predict_image(image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return CLASS_NAMES[predicted.item()], confidence.item() * 100

# Upload
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=400)
        label, conf = predict_image(img)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf:.2f}%")

# Webcam
if option == "Use Webcam":
    cam = st.camera_input("Take Picture")
    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img, width=400)
        label, conf = predict_image(img)
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf:.2f}%")
