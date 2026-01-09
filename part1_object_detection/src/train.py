import torch
from torch.utils.data import DataLoader
from dataset import Coco5Dataset
from model import SimpleDetector
from loss import DetectionLoss

DEVICE = "cpu"
EPOCHS = 20
BATCH_SIZE = 8

def collate_fn(batch):
    """Custom collate to handle varying number of boxes per image"""
    imgs, boxes = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, boxes

dataset = Coco5Dataset("../../datasets/coco5", split="train")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = SimpleDetector().to(DEVICE)
criterion = DetectionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, boxes in loader:
        # Take first object from each image in batch
        batch_targets = []
        for box_list in boxes:
            if len(box_list) > 0:
                batch_targets.append(box_list[0])
            else:
                # Skip images with no boxes
                continue
        
        if len(batch_targets) == 0:
            continue
            
        targets = torch.stack(batch_targets, 0)
        imgs = imgs[:len(batch_targets)].to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "detector.pth")
print("✅ Model saved as detector.pth")
