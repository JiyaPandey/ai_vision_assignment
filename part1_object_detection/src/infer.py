import cv2
import os
import random
import torch
from model import SimpleDetector

CLASSES = ["person", "car", "dog", "bottle", "chair"]

model = SimpleDetector()
model.load_state_dict(torch.load("detector.pth"))
model.eval()

# Load specific image with a dog
img = cv2.imread("../../datasets/coco5/images/train/000000000074.jpg")
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
