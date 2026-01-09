import cv2
import os
import torch
from model import SimpleDetector

CLASSES = ["person", "car", "dog", "bottle", "chair"]

model = SimpleDetector()
model.load_state_dict(torch.load("detector.pth"))
model.eval()

val_dir = "../../datasets/coco5/images/val"
label_dir = "../../datasets/coco5/labels/val"

correct = 0
total = 0

for img_file in os.listdir(val_dir):  # Test on ALL images
    img_path = os.path.join(val_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    inp = cv2.resize(img, (224, 224))
    inp = torch.tensor(inp / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    
    with torch.no_grad():
        out = model(inp)[0]
    
    pred_cls = out[4:].argmax().item()
    
    # Read ground truth
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if lines:
            gt_cls = int(lines[0].split()[0])
            total += 1
            if pred_cls == gt_cls:
                correct += 1
                print(f"✅ {img_file}: Predicted={CLASSES[pred_cls]}, GT={CLASSES[gt_cls]}")
            else:
                print(f"❌ {img_file}: Predicted={CLASSES[pred_cls]}, GT={CLASSES[gt_cls]}")

print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")
