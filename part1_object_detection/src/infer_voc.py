import cv2
import os
import random
import torch
from model import ImprovedDetector

CLASSES = ["person", "car", "dog", "cat", "bicycle"]

model = ImprovedDetector(num_classes=5)
model.load_state_dict(torch.load("detector_voc_best.pth"))
model.eval()

# Pick a random validation image from VOC
val_dir = os.path.expanduser("~/ai_vision_assignment/datasets/VOCdevkit/VOC2007/JPEGImages")
val_split = os.path.expanduser("~/ai_vision_assignment/datasets/VOCdevkit/VOC2007/ImageSets/Main/val.txt")

with open(val_split, 'r') as f:
    val_ids = [line.strip() for line in f.readlines()]

random_id = random.choice(val_ids)
img_path = os.path.join(val_dir, f"{random_id}.jpg")

print(f"Testing on: {random_id}.jpg")

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

cv2.imwrite("output_voc.jpg", img)
print(f"✅ Output saved as output_voc.jpg")
print(f"🎯 Detected: {CLASSES[cls]}")
