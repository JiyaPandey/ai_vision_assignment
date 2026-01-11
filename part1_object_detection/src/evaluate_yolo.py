"""
Evaluate model performance on YOLO dataset
"""
import torch
from torch.utils.data import DataLoader
from yolo_dataset import YOLODataset
from model import ImprovedDetector
from loss import ImprovedDetectionLoss
import os
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# Dataset paths
DATASET_ROOT = os.path.expanduser("~/ai_vision_assignment/part1_object_detection/5 class dataset.v1i.yolov8")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test/images")
TEST_LABEL_DIR = os.path.join(DATASET_ROOT, "test/labels")

CLASSES = ['Ipad', 'backpack', 'hand', 'phone', 'wallet']

def compute_metrics(model, test_loader):
    model.eval()
    all_ious = []
    per_class_correct = {i: 0 for i in range(5)}
    per_class_total = {i: 0 for i in range(5)}
    
    # For precision, recall, and F1 score
    per_class_tp = {i: 0 for i in range(5)}  # True positives
    per_class_fp = {i: 0 for i in range(5)}  # False positives
    per_class_fn = {i: 0 for i in range(5)}  # False negatives
    
    # For mAP calculation
    all_detections = []  # (class_id, confidence, iou, gt_class)
    
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            preds = model(imgs)
            
            # Classification metrics
            pred_cls = preds[:, 4:].argmax(dim=1)
            pred_conf = torch.softmax(preds[:, 4:], dim=1).max(dim=1)[0]  # confidence scores
            gt_cls = targets[:, 0].long()
            
            # IoU metrics
            pred_box = preds[:, :4]
            gt_box = targets[:, 1:]
            
            pred_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
            pred_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
            pred_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
            pred_y2 = pred_box[:, 1] + pred_box[:, 3] / 2
            
            gt_x1 = gt_box[:, 0] - gt_box[:, 2] / 2
            gt_y1 = gt_box[:, 1] - gt_box[:, 3] / 2
            gt_x2 = gt_box[:, 0] + gt_box[:, 2] / 2
            gt_y2 = gt_box[:, 1] + gt_box[:, 3] / 2
            
            inter_x1 = torch.max(pred_x1, gt_x1)
            inter_y1 = torch.max(pred_y1, gt_y1)
            inter_x2 = torch.min(pred_x2, gt_x2)
            inter_y2 = torch.min(pred_y2, gt_y2)
            
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
            
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            union_area = pred_area + gt_area - inter_area
            
            iou = inter_area / (union_area + 1e-6)
            all_ious.extend(iou.cpu().numpy())
            
            # Compute per-class metrics
            for i in range(len(gt_cls)):
                cls_id = gt_cls[i].item()
                pred_id = pred_cls[i].item()
                iou_val = iou[i].item()
                conf = pred_conf[i].item()
                
                per_class_total[cls_id] += 1
                
                # Store detection info for mAP
                all_detections.append((pred_id, conf, iou_val, cls_id))
                
                # Consider a detection correct if IoU > 0.5 and class matches
                if iou_val > 0.5 and pred_id == cls_id:
                    per_class_correct[cls_id] += 1
                    per_class_tp[cls_id] += 1
                else:
                    # False negative for the ground truth class
                    per_class_fn[cls_id] += 1
                    # False positive for the predicted class (if different)
                    if pred_id != cls_id:
                        per_class_fp[pred_id] += 1
    
    return all_ious, per_class_correct, per_class_total, per_class_tp, per_class_fp, per_class_fn, all_detections

def main():
    print(f"Loading test dataset from {DATASET_ROOT}")
    test_dataset = YOLODataset(TEST_IMG_DIR, TEST_LABEL_DIR, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load model
    model = ImprovedDetector(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load("detector_yolo_best.pth", map_location=DEVICE))
    
    print(f"Evaluating on {len(test_dataset)} test images...")
    all_ious, per_class_correct, per_class_total, per_class_tp, per_class_fp, per_class_fn, all_detections = compute_metrics(model, test_loader)
    
    # Overall metrics
    mean_iou = np.mean(all_ious)
    total_correct = sum(per_class_correct.values())
    total_samples = sum(per_class_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Calculate mAP@0.5 (mean Average Precision at IoU threshold 0.5)
    per_class_ap = {}
    for cls_id in range(5):
        # Get all detections for this class
        cls_detections = [(conf, iou, gt) for pred, conf, iou, gt in all_detections if pred == cls_id]
        if len(cls_detections) == 0:
            per_class_ap[cls_id] = 0.0
            continue
        
        # Sort by confidence
        cls_detections.sort(key=lambda x: x[0], reverse=True)
        
        tp = 0
        fp = 0
        total_gt = per_class_total[cls_id]
        
        precisions = []
        recalls = []
        
        for conf, iou, gt in cls_detections:
            if iou > 0.5 and gt == cls_id:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_gt if total_gt > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for r_threshold in np.linspace(0, 1, 11):
            precisions_above = [p for p, r in zip(precisions, recalls) if r >= r_threshold]
            if len(precisions_above) > 0:
                ap += max(precisions_above) / 11
        
        per_class_ap[cls_id] = ap
    
    mAP = np.mean(list(per_class_ap.values()))
    
    # Calculate overall precision, recall, F1 score
    total_tp = sum(per_class_tp.values())
    total_fp = sum(per_class_fp.values())
    total_fn = sum(per_class_fn.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Mean IoU: {mean_iou:.3f}")
    print(f"mAP@0.5: {mAP:.3f}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1 Score: {overall_f1:.3f}")
    print("\nPer-Class Accuracy:")
    for i in range(5):
        acc = per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        print(f"  {CLASSES[i]:12s}: {acc:.3f} ({per_class_correct[i]}/{per_class_total[i]})")
    print("\nPer-Class Average Precision (AP@0.5):")
    for i in range(5):
        print(f"  {CLASSES[i]:12s}: {per_class_ap[i]:.3f}")
    print("\nPer-Class Precision, Recall, F1:")
    for i in range(5):
        prec = per_class_tp[i] / (per_class_tp[i] + per_class_fp[i]) if (per_class_tp[i] + per_class_fp[i]) > 0 else 0
        rec = per_class_tp[i] / (per_class_tp[i] + per_class_fn[i]) if (per_class_tp[i] + per_class_fn[i]) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {CLASSES[i]:12s}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    print("="*70)

if __name__ == "__main__":
    main()
