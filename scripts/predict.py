import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pyttsx3
from torchvision import transforms, models
from PIL import Image

# ---------------- CONFIG ----------------
NUM_CLASSES = 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_names = [
    "call","dislike","fist","four","like","mute",
    "ok","one","palm","peace","peace_inverted",
    "rock","stop","stop_inverted","three",
    "three2","two_up","two_up_inverted"
]

CONFIDENCE_THRESHOLD = 0.60   # adjust if needed

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_spoken = ""

# ---------------- LOAD MODEL ----------------
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(torch.load("model/hand_sign_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    conf = confidence.item()
    label = class_names[pred.item()]

    if conf < CONFIDENCE_THRESHOLD:
        label = "No Sign"

    display_text = f"{label} ({conf:.2f})"

    # Show prediction
    cv2.putText(frame, display_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    # Speak only when new valid sign detected
    if label != last_spoken and label != "No Sign":
        engine.say(label)
        engine.runAndWait()
        last_spoken = label

    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
