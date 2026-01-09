import os
import shutil
import random

# Paths
BASE_DIR = "coco5"
IMG_DIR = os.path.join(BASE_DIR, "images", "train")
LBL_DIR = os.path.join(BASE_DIR, "labels", "train")

VAL_IMG_DIR = os.path.join(BASE_DIR, "images", "val")
VAL_LBL_DIR = os.path.join(BASE_DIR, "labels", "val")

# Create val directories
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LBL_DIR, exist_ok=True)

# Get all images
images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
random.shuffle(images)

# 20% for validation
val_count = int(0.2 * len(images))
val_images = images[:val_count]

for img in val_images:
    label = img.replace(".jpg", ".txt")

    shutil.move(
        os.path.join(IMG_DIR, img),
        os.path.join(VAL_IMG_DIR, img)
    )

    shutil.move(
        os.path.join(LBL_DIR, label),
        os.path.join(VAL_LBL_DIR, label)
    )

print("✅ Dataset split complete")
print(f"Validation images: {len(val_images)}")
print(f"Training images left: {len(images) - len(val_images)}")

