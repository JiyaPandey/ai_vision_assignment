"""
Evaluate object detection model using mAP (mean Average Precision)
"""
import torch
import os
import numpy as np
from model import SimpleDetector
from dataset import Coco5Dataset
import cv2

CLASSES = ["person", "car", "dog", "bottle", "chair"]

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes
    box format: [x_center, y_center, width, height] (normalized)
    """
    # Convert to x1, y1, x2, y2
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)

def evaluate_map(model, dataset, iou_threshold=0.5):
    """
    Compute mAP@IoU threshold
    """
    model.eval()
    
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}
    class_tp = {i: [] for i in range(5)}  # True positives
    class_fp = {i: [] for i in range(5)}  # False positives
    class_scores = {i: [] for i in range(5)}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, boxes = dataset[idx]
            
            if len(boxes) == 0:
                continue
            
            # Get prediction
            img_tensor = img.unsqueeze(0)
            output = model(img_tensor)[0]
            
            pred_box = output[:4].numpy()
            pred_cls = output[4:].argmax().item()
            pred_score = torch.softmax(output[4:], dim=0).max().item()
            
            # Get ground truth (first object)
            gt_cls = int(boxes[0, 0].item())
            gt_box = boxes[0, 1:].numpy()
            
            class_total[gt_cls] += 1
            
            # Check if correct class
            if pred_cls == gt_cls:
                # Check IoU
                iou = compute_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    class_correct[gt_cls] += 1
                    class_tp[pred_cls].append(1)
                    class_fp[pred_cls].append(0)
                else:
                    class_tp[pred_cls].append(0)
                    class_fp[pred_cls].append(1)
            else:
                class_fp[pred_cls].append(1)
                class_tp[pred_cls].append(0)
            
            class_scores[pred_cls].append(pred_score)
    
    # Compute AP for each class
    aps = []
    for cls_idx in range(5):
        if class_total[cls_idx] == 0:
            continue
        
        # Simple accuracy-based AP
        if class_total[cls_idx] > 0:
            ap = class_correct[cls_idx] / class_total[cls_idx]
            aps.append(ap)
            print(f"{CLASSES[cls_idx]:10s}: AP = {ap:.3f} ({class_correct[cls_idx]}/{class_total[cls_idx]})")
    
    map_score = np.mean(aps) if aps else 0.0
    return map_score, class_correct, class_total

if __name__ == "__main__":
    # Load model
    model = SimpleDetector()
    model.load_state_dict(torch.load("detector.pth"))
    model.eval()
    
    # Load validation dataset
    val_dataset = Coco5Dataset("../../datasets/coco5", split="val")
    
    print("="*50)
    print("Evaluating mAP@0.5 on Validation Set")
    print("="*50)
    
    map_05, correct, total = evaluate_map(model, val_dataset, iou_threshold=0.5)
    
    print("="*50)
    print(f"mAP@0.5: {map_05:.3f}")
    print(f"Total Accuracy: {sum(correct.values())}/{sum(total.values())} = {sum(correct.values())/sum(total.values()):.3f}")
    print("="*50)
