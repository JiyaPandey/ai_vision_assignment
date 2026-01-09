"""
Improved training script with better techniques for object detection
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Coco5Dataset
from model import SimpleDetector
from loss import DetectionLoss
import os
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200  # More epochs for better convergence
BATCH_SIZE = 8  # Larger batch size
LEARNING_RATE = 5e-4  # Lower learning rate for stability
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def collate_fn(batch):
    """Custom collate to handle varying number of boxes per image"""
    imgs, boxes = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, boxes

# Load datasets
train_dataset = Coco5Dataset("../../datasets/coco5", split="train", augment=True)
val_dataset = Coco5Dataset("../../datasets/coco5", split="val", augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize model
model = SimpleDetector().to(DEVICE)
criterion = DetectionLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    box1_x1 = box1[:, 0] - box1[:, 2] / 2
    box1_y1 = box1[:, 1] - box1[:, 3] / 2
    box1_x2 = box1[:, 0] + box1[:, 2] / 2
    box1_y2 = box1[:, 1] + box1[:, 3] / 2
    
    box2_x1 = box2[:, 0] - box2[:, 2] / 2
    box2_y1 = box2[:, 1] - box2[:, 3] / 2
    box2_x2 = box2[:, 0] + box2[:, 2] / 2
    box2_y2 = box2[:, 1] + box2[:, 3] / 2
    
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)

def validate(model, val_loader):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_iou = 0
    
    with torch.no_grad():
        for imgs, boxes in val_loader:
            batch_targets = []
            for box_list in boxes:
                if len(box_list) > 0:
                    batch_targets.append(box_list[0])
            
            if len(batch_targets) == 0:
                continue
            
            targets = torch.stack(batch_targets, 0)
            imgs = imgs[:len(batch_targets)].to(DEVICE)
            targets = targets.to(DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            
            # Compute accuracy
            pred_cls = preds[:, 4:].argmax(dim=1)
            gt_cls = targets[:, 0].long()
            correct += (pred_cls == gt_cls).sum().item()
            total += len(targets)
            
            # Compute IoU
            pred_box = preds[:, :4]
            gt_box = targets[:, 1:]
            iou = compute_iou(pred_box, gt_box)
            total_iou += iou.sum().item()
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    avg_iou = total_iou / total if total > 0 else 0
    
    return avg_loss, accuracy, avg_iou

# Training loop
best_val_loss = float('inf')
best_val_iou = 0

print(f"Training on {DEVICE}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print("="*70)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Training
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for imgs, boxes in pbar:
        # Take first object from each image in batch
        batch_targets = []
        for box_list in boxes:
            if len(box_list) > 0:
                batch_targets.append(box_list[0])
        
        if len(batch_targets) == 0:
            continue
        
        targets = torch.stack(batch_targets, 0)
        imgs = imgs[:len(batch_targets)].to(DEVICE)
        targets = targets.to(DEVICE)
        
        preds = model(imgs)
        loss = criterion(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # Validation
    val_loss, val_acc, val_iou = validate(model, val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.3f} | "
          f"Val IoU: {val_iou:.3f}")
    
    # Save best model based on validation IoU
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "detector_best.pth")
        print(f"  ✅ Saved best model (IoU: {val_iou:.3f})")
    
    # Save checkpoint every 20 epochs
    if (epoch + 1) % 20 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"detector_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
        }, checkpoint_path)

# Save final model
torch.save(model.state_dict(), "detector.pth")
print("\n" + "="*70)
print("✅ Training completed!")
print(f"✅ Best validation IoU: {best_val_iou:.3f}")
print(f"✅ Final model saved as detector.pth")
print(f"✅ Best model saved as detector_best.pth")
