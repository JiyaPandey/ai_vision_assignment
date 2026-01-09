"""
Debug script to understand why model has 0% accuracy
"""
import torch
import numpy as np
from model import SimpleDetector
from dataset import Coco5Dataset
import cv2

CLASSES = ["person", "car", "dog", "bottle", "chair"]

# Load model
model = SimpleDetector()
model.load_state_dict(torch.load("detector.pth"))
model.eval()

# Load validation dataset
val_dataset = Coco5Dataset("../../datasets/coco5", split="val")

print("="*60)
print("DEBUGGING MODEL PREDICTIONS")
print("="*60)

# Check a few samples
for idx in range(min(5, len(val_dataset))):
    img, boxes = val_dataset[idx]
    
    if len(boxes) == 0:
        continue
    
    # Get prediction
    img_tensor = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)[0]
    
    pred_box = output[:4].detach().numpy()
    pred_cls_logits = output[4:].detach().numpy()
    pred_cls = output[4:].argmax().item()
    pred_probs = torch.softmax(output[4:], dim=0).detach().numpy()
    
    # Ground truth
    gt_cls = int(boxes[0, 0].item())
    gt_box = boxes[0, 1:].numpy()
    
    print(f"\nSample {idx}:")
    print(f"  Ground Truth: {CLASSES[gt_cls]}")
    print(f"  GT Box (norm): [{gt_box[0]:.3f}, {gt_box[1]:.3f}, {gt_box[2]:.3f}, {gt_box[3]:.3f}]")
    print(f"  Predicted: {CLASSES[pred_cls]}")
    print(f"  Pred Box (norm): [{pred_box[0]:.3f}, {pred_box[1]:.3f}, {pred_box[2]:.3f}, {pred_box[3]:.3f}]")
    print(f"  Class Logits: {pred_cls_logits}")
    print(f"  Class Probs: {pred_probs}")
    print(f"  Confidence: {pred_probs[pred_cls]:.3f}")
    
    # Check if box coordinates are reasonable
    if any(pred_box < 0) or any(pred_box > 1):
        print(f"  ⚠️  WARNING: Box coordinates out of [0,1] range!")
    
    # Compute IoU
    def compute_iou(box1, box2):
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2
        
        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2
        
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    iou = compute_iou(pred_box, gt_box)
    print(f"  IoU: {iou:.3f}")
    print(f"  Class Match: {pred_cls == gt_cls}")
    print(f"  Detection OK (IoU>0.5 & class match): {iou >= 0.5 and pred_cls == gt_cls}")

print("\n" + "="*60)
print("CHECKING MODEL WEIGHTS")
print("="*60)

# Check if model weights are reasonable
for name, param in model.named_parameters():
    print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, min={param.data.min():.4f}, max={param.data.max():.4f}")

print("\n" + "="*60)
