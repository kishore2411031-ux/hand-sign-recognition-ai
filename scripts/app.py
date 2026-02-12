import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "model/hand_sign_model.pth"
DATA_DIR = "data/train"   # change if needed
DEVICE = "cpu"

# ---------------- LOAD CLASS NAMES ----------------
if os.path.exists(DATA_DIR):
    CLASS_NAMES = sorted(os.listdir(DATA_DIR))
else:
    # fallback manual class names
    CLASS_NAMES = ["A","B","C","D","E","F","G","H","I","J",
                   "K","L","M","N","O","P","Q","R"]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- PREDICT FUNCTION ----------------
def predict_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_class = torch.max(probs, 0)
    return CLASS_NAMES[pred_class], float(confidence)*100


# ---------------- UI ----------------
st.set_page_config(page_title="AI Hand Sign Recognition", layout="centered")

st.title("🤖 AI Hand Sign Recognition")
st.write("Upload an image or use webcam to detect hand sign")

option = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

image = None

# ---------- Upload ----------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# ---------- Webcam ----------
if option == "Use Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

# ---------- Prediction ----------
if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    label, confidence = predict_image(image)

    st.success(f"✅ Prediction: {label}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    if confidence < 50:
        st.warning("⚠ Low confidence — Try clearer hand sign")

