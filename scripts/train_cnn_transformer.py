import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ---------------- CONFIG ----------------
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoint.pth"

# ------------- TRANSFORMS ---------------
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ------------- DATASET ------------------
train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ------------- MODEL --------------------
cnn = models.mobilenet_v2(pretrained=True)
cnn.classifier[1] = nn.Linear(cnn.last_channel, NUM_CLASSES)
cnn = cnn.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)

# ------------- CHECKPOINT LOAD ----------
start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print("🔄 Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    cnn.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resuming from epoch {start_epoch}")

# ------------- TRAIN --------------------
print("🚀 Training started...")

for epoch in range(start_epoch, EPOCHS):
    cnn.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = cnn(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

    # Save checkpoint after each epoch
    torch.save({
        "epoch": epoch,
        "model_state": cnn.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, CHECKPOINT_PATH)

    print("💾 Checkpoint saved")

# ------------- FINAL SAVE ----------------
os.makedirs("model", exist_ok=True)
torch.save(cnn.state_dict(), "model/hand_sign_model.pth")

# Remove checkpoint after full training (optional)
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print("🎉 Training completed & model saved")
