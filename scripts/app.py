import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "model/hand_sign_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------- LOAD MODEL ---------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # If model was saved with class mapping
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
        state_dict = checkpoint["model_state"]
    else:
        # Fallback (IMPORTANT: change order if needed)
        class_names = [
            "Like",
            "Fist",
            "One",
            "Two",
            "Three",
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
            "ThumbsDown"
        ]
        state_dict = checkpoint

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model()

# ------------- UI -----------------------
st.set_page_config(page_title="AI Hand Sign Recognition", layout="centered")
st.title("🤖 AI Hand Sign Recognition")
st.write("Upload image or use webcam to detect hand sign")

option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

def predict_image(image):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100

    return predicted_label, confidence_score

# -------- Upload Image --------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Selected Image", width=500)

        label, conf = predict_image(image)

        st.success(f"✅ Prediction: {label}")
        st.info(f"📊 Confidence: {conf:.2f}%")

# -------- Webcam --------
elif option == "Use Webcam":
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", width=500)

        label, conf = predict_image(image)

        st.success(f"✅ Prediction: {label}")
        st.info(f"📊 Confidence: {conf:.2f}%")
