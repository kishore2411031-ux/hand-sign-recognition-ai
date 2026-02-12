import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "hand_sign_model.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "train")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 18

# Automatically read class names from dataset folder
CLASS_NAMES = sorted(os.listdir(DATA_DIR))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
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

# ---------------- PREDICT FUNCTION ----------------
def predict_image(image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return CLASS_NAMES[predicted.item()], confidence.item()

# ---------------- UI ----------------
st.set_page_config(page_title="Hand Sign Recognition", layout="centered")
st.title("✋ Hand Sign Recognition System")

option = st.radio("Choose Input Method:", ["Upload Image", "Use Webcam"])

# ================= UPLOAD =================
if option == "Upload Image":

    uploaded_file = st.file_uploader("Upload Hand Sign Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        label, confidence = predict_image(image)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence*100:.2f}%")

# ================= WEBCAM =================
elif option == "Use Webcam":

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            label, confidence = predict_image(pil_image)

            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
