import cv2
import os
import torch
from model import SimpleDetector

CLASSES = ["person", "car", "dog", "bottle", "chair"]

model = SimpleDetector()
model.load_state_dict(torch.load("detector.pth"))
model.eval()

# Load specific image with a dog
img_path = "../../datasets/coco5/images/train/000000000074.jpg"
label_path = "../../datasets/coco5/labels/train/000000000074.txt"

img = cv2.imread(img_path)
h, w, _ = img.shape

inp = cv2.resize(img, (224, 224))
inp = torch.tensor(inp / 255.0).permute(2, 0, 1).unsqueeze(0).float()

with torch.no_grad():
    out = model(inp)[0]

# Predicted box
pred_x, pred_y, pred_bw, pred_bh = out[:4].numpy()
pred_cls = out[4:].argmax().item()

# Ground truth box (first dog)
with open(label_path, 'r') as f:
    gt_line = f.readlines()[0].strip().split()
    gt_cls, gt_x, gt_y, gt_bw, gt_bh = map(float, gt_line)

# Convert normalized → pixels for PREDICTION
pred_cx, pred_cy = int(pred_x * w), int(pred_y * h)
pred_w_px, pred_h_px = int(pred_bw * w), int(pred_bh * h)
pred_x1, pred_y1 = pred_cx - pred_w_px // 2, pred_cy - pred_h_px // 2
pred_x2, pred_y2 = pred_cx + pred_w_px // 2, pred_cy + pred_h_px // 2

# Convert normalized → pixels for GROUND TRUTH
gt_cx, gt_cy = int(gt_x * w), int(gt_y * h)
gt_w_px, gt_h_px = int(gt_bw * w), int(gt_bh * h)
gt_x1, gt_y1 = gt_cx - gt_w_px // 2, gt_cy - gt_h_px // 2
gt_x2, gt_y2 = gt_cx + gt_w_px // 2, gt_cy + gt_h_px // 2

# Draw ground truth in BLUE
cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), (255, 0, 0), 2)
cv2.putText(img, f"GT: {CLASSES[int(gt_cls)]}", (gt_x1, gt_y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Draw prediction in GREEN
cv2.rectangle(img, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 2)
cv2.putText(img, f"Pred: {CLASSES[pred_cls]}", (pred_x1, pred_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("output_comparison.jpg", img)
print(f"✅ Output saved as output_comparison.jpg")
print(f"🎯 Predicted: {CLASSES[pred_cls]} at ({pred_x:.3f}, {pred_y:.3f}, {pred_bw:.3f}, {pred_bh:.3f})")
print(f"📍 Ground Truth: {CLASSES[int(gt_cls)]} at ({gt_x:.3f}, {gt_y:.3f}, {gt_bw:.3f}, {gt_bh:.3f})")
