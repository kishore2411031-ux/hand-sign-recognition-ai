import os
import random
import shutil

SRC = r"C:\project\archive\hagrid-classification-512p"
DST = r"C:\project\data"

TRAIN_RATIO = 0.8
random.seed(42)

for split in ["train", "val"]:
    for cls in os.listdir(SRC):
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

for cls in os.listdir(SRC):
    cls_path = os.path.join(SRC, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path)
              if f.lower().endswith(('.jpg','.png','.jpeg'))]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy2(
            os.path.join(cls_path, img),
            os.path.join(DST, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy2(
            os.path.join(cls_path, img),
            os.path.join(DST, "val", cls, img)
        )

print("Dataset split completed.")
