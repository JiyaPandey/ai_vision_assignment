import cv2
import os
import random
import torch
from model import ImprovedDetector

CLASSES = ['Ipad', 'backpack', 'hand', 'phone', 'wallet']

# Load model
model = ImprovedDetector(num_classes=5)
model.load_state_dict(torch.load("detector_yolo_best.pth"))
model.eval()

# Dataset paths
DATASET_ROOT = os.path.expanduser("~/ai_vision_assignment/part1_object_detection/5 class dataset.v1i.yolov8")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "valid/images")

# Get all validation images
val_images = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if len(val_images) == 0:
    print("No validation images found!")
    exit(1)

# Pick a random validation image
random_img = random.choice(val_images)
img_path = os.path.join(VAL_IMG_DIR, random_img)

print(f"Testing on: {random_img}")

img = cv2.imread(img_path)
h, w, _ = img.shape

# Prepare input with ImageNet normalization
inp = cv2.resize(img, (224, 224))
inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
inp = inp / 255.0
# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inp = (inp - mean) / std
inp = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

# Run inference
with torch.no_grad():
    out = model(inp)[0]

# Parse output
x, y, bw, bh = out[:4]
cls = out[4:].argmax().item()

# Convert normalized coordinates to pixels
cx, cy = int(x * w), int(y * h)
bw, bh = int(bw * w), int(bh * h)

x1, y1 = cx - bw // 2, cy - bh // 2
x2, y2 = cx + bw // 2, cy + bh // 2

# Draw bounding box and label
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, CLASSES[cls], (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save output
output_path = "output_yolo.jpg"
cv2.imwrite(output_path, img)
print(f"✅ Output saved as {output_path}")
print(f"🎯 Detected: {CLASSES[cls]}")
