"""
Demo script to visualize detection results on random images
"""
import cv2
import os
import random
import torch
from model import ImprovedDetector
import time

CLASSES = ['Ipad', 'backpack', 'hand', 'phone', 'wallet']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "detector_yolo_best.pth")
model = ImprovedDetector(num_classes=5, pretrained=False)  # pretrained=False for inference
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Dataset paths
DEMO_IMG_DIR = os.path.expanduser("~/ai_vision_assignment/part1_object_detection/demoimg")
DATASET_ROOT = os.path.expanduser("~/ai_vision_assignment/part1_object_detection/5 class dataset.v1i.yolov8")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "valid/images")

# Check if demoimg folder exists and has images
if os.path.exists(DEMO_IMG_DIR):
    demo_images = [f for f in os.listdir(DEMO_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(demo_images) > 0:
        print(f"✨ Using {len(demo_images)} images from demoimg folder")
        selected_images = demo_images
        IMG_DIR = DEMO_IMG_DIR
    else:
        print("⚠️  demoimg folder is empty, using validation images instead")
        val_images = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(val_images) == 0:
            print("No validation images found!")
            exit(1)
        num_images = min(15, len(val_images))
        selected_images = random.sample(val_images, num_images)
        IMG_DIR = VAL_IMG_DIR
else:
    print("📁 demoimg folder not found, using validation images instead")
    val_images = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(val_images) == 0:
        print("No validation images found!")
        exit(1)
    num_images = min(15, len(val_images))
    selected_images = random.sample(val_images, num_images)
    IMG_DIR = VAL_IMG_DIR

num_images = len(selected_images)

print(f"Running detection on {num_images} random images...")
print("Press 'q' to skip to next image or wait 3 seconds for auto-advance\n")

for idx, img_name in enumerate(selected_images, 1):
    img_path = os.path.join(IMG_DIR, img_name)
    
    print(f"[{idx}/{num_images}] Processing: {img_name}")
    
    # Read and prepare image
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠️  Failed to load image, skipping...")
        continue
    
    h, w, _ = img.shape
    
    # Prepare input with ImageNet normalization
    inp = cv2.resize(img, (224, 224))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp / 255.0
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp = (inp - mean) / std
    inp = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        out = model(inp)[0]
    
    # Parse output
    x, y, bw, bh = out[:4].cpu()
    cls = out[4:].argmax().item()
    
    # Convert normalized coordinates to pixels
    cx, cy = int(x * w), int(y * h)
    bw, bh = int(bw * w), int(bh * h)
    
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    
    # Draw bounding box and label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{CLASSES[cls]}"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add image counter in the corner
    counter_text = f"Image {idx}/{num_images}"
    cv2.putText(img, counter_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    print(f"  🎯 Detected: {CLASSES[cls]}")
    
    # Display image
    cv2.imshow('YOLO Detection Demo', img)
    
    # Wait for 3 seconds or until 'q' is pressed
    key = cv2.waitKey(3000)  # 3000 ms = 3 seconds
    
    if key == ord('q') or key == 27:  # 'q' or ESC to quit
        print("\nDemo stopped by user.")
        break

cv2.destroyAllWindows()
print("\n✅ Demo completed!")
