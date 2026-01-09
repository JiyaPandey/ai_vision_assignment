"""
Train with Pascal VOC 2007 dataset (5 classes)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from voc_dataset import PascalVOCDataset
from model import ImprovedDetector
from loss import ImprovedDetectionLoss
import os
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PATIENCE = 15
VOC_ROOT = os.path.expanduser("~/ai_vision_assignment/datasets/VOCdevkit/VOC2007")

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_iou = 0
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            
            # Classification accuracy
            pred_cls = preds[:, 4:].argmax(dim=1)
            gt_cls = targets[:, 0].long()
            correct += (pred_cls == gt_cls).sum().item()
            total += len(targets)
            
            # Compute IoU
            pred_box = preds[:, :4]
            gt_box = targets[:, 1:]
            
            # IoU calculation
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
            total_iou += iou.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0
    avg_iou = total_iou / total if total > 0 else 0
    
    return avg_loss, accuracy, avg_iou

def main():
    # Load datasets
    print(f"Loading Pascal VOC 2007 dataset from {VOC_ROOT}")
    train_dataset = PascalVOCDataset(VOC_ROOT, split="train", augment=True)
    val_dataset = PascalVOCDataset(VOC_ROOT, split="val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = ImprovedDetector(num_classes=5).to(DEVICE)
    criterion = ImprovedDetectionLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    best_val_iou = 0
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0

    print(f"Training on {DEVICE}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print("="*70)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Validation
        val_loss, val_acc, val_iou = validate(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"Val IoU: {val_iou:.3f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "detector_voc_best.pth")
            print(f"  ✅ Saved best model (IoU: {val_iou:.3f}, Acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Best model from epoch {best_epoch} with IoU: {best_val_iou:.3f}")
                break
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"detector_voc_epoch{epoch+1}.pth")

    print("\n" + "="*70)
    print("✅ Training completed!")
    print(f"✅ Best validation IoU: {best_val_iou:.3f}")
    print(f"✅ Best validation Acc: {best_val_acc:.3f}")
    torch.save(model.state_dict(), "detector_voc_final.pth")

if __name__ == "__main__":
    main()
