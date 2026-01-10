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
    
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            preds = model(imgs)
            
            # Classification metrics
            pred_cls = preds[:, 4:].argmax(dim=1)
            gt_cls = targets[:, 0].long()
            
            for i in range(len(gt_cls)):
                cls_id = gt_cls[i].item()
                per_class_total[cls_id] += 1
                if pred_cls[i] == gt_cls[i]:
                    per_class_correct[cls_id] += 1
            
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
    
    return all_ious, per_class_correct, per_class_total

def main():
    print(f"Loading test dataset from {DATASET_ROOT}")
    test_dataset = YOLODataset(TEST_IMG_DIR, TEST_LABEL_DIR, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load model
    model = ImprovedDetector(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load("detector_yolo_best.pth"))
    
    print(f"Evaluating on {len(test_dataset)} test images...")
    all_ious, per_class_correct, per_class_total = compute_metrics(model, test_loader)
    
    # Overall metrics
    mean_iou = np.mean(all_ious)
    total_correct = sum(per_class_correct.values())
    total_samples = sum(per_class_total.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Mean IoU: {mean_iou:.3f}")
    print("\nPer-Class Accuracy:")
    for i in range(5):
        acc = per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        print(f"  {CLASSES[i]:12s}: {acc:.3f} ({per_class_correct[i]}/{per_class_total[i]})")
    print("="*70)

if __name__ == "__main__":
    main()
