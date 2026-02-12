import os
import shutil
import random

SRC_DIR = r"C:\project\archive\hagrid-classification-512p"
DST_DIR = r"C:\project\data"

TRAIN_PER_CLASS = 400
VAL_PER_CLASS = 100

random.seed(42)

for split in ["train", "val"]:
    os.makedirs(os.path.join(DST_DIR, split), exist_ok=True)

classes = os.listdir(SRC_DIR)

for cls in classes:
    src_cls = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(src_cls):
        continue

    images = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    random.shuffle(images)

    train_imgs = images[:TRAIN_PER_CLASS]
    val_imgs = images[TRAIN_PER_CLASS:TRAIN_PER_CLASS+VAL_PER_CLASS]

    train_dst = os.path.join(DST_DIR, "train", cls)
    val_dst = os.path.join(DST_DIR, "val", cls)

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(src_cls, img), os.path.join(train_dst, img))

    for img in val_imgs:
        shutil.copy(os.path.join(src_cls, img), os.path.join(val_dst, img))

    print(f"{cls}: train={len(train_imgs)}, val={len(val_imgs)}")

print("✅ Small dataset created successfully")
