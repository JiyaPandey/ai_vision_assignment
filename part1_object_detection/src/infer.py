import cv2
import os
import random
import torch
from model import ImprovedDetector

CLASSES = ["person", "car", "dog", "bottle", "chair"]

model = ImprovedDetector()
model.load_state_dict(torch.load("detector_improved_best.pth"))
model.eval()

# Pick a random validation image
val_dir = os.path.join(os.path.dirname(__file__), "../../datasets/coco5/images/val")
val_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
random_img = random.choice(val_images)

img_path = os.path.join(val_dir, random_img)
print(f"Testing on: {random_img}")

img = cv2.imread(img_path)
h, w, _ = img.shape

inp = cv2.resize(img, (224, 224))
inp = torch.tensor(inp / 255.0).permute(2, 0, 1).unsqueeze(0).float()

with torch.no_grad():
    out = model(inp)[0]

x, y, bw, bh = out[:4]
cls = out[4:].argmax().item()

# convert normalized → pixels
cx, cy = int(x * w), int(y * h)
bw, bh = int(bw * w), int(bh * h)

x1, y1 = cx - bw // 2, cy - bh // 2
x2, y2 = cx + bw // 2, cy + bh // 2

cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, CLASSES[cls], (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("output.jpg", img)
print(f"✅ Output saved as output.jpg")
print(f"🎯 Detected: {CLASSES[cls]}")
