"""
Measure inference speed (FPS - Frames Per Second)
"""
import torch
import cv2
import time
import numpy as np
from model import SimpleDetector
import os

def measure_fps(model, num_images=100, img_size=224):
    """
    Measure inference speed in FPS
    """
    model.eval()
    
    # Create dummy images
    dummy_imgs = torch.randn(num_images, 3, img_size, img_size)
    
    # Warmup
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_imgs[i:i+1])
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for i in range(num_images):
            _ = model(dummy_imgs[i:i+1])
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_images / total_time
    avg_time_ms = (total_time / num_images) * 1000
    
    return fps, avg_time_ms

def measure_real_image_fps(model, image_dir, num_images=50):
    """
    Measure FPS on real images
    """
    model.eval()
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:num_images]
    
    times = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float()
        
        start = time.time()
        with torch.no_grad():
            _ = model(img_tensor)
        end = time.time()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    return fps, avg_time * 1000

if __name__ == "__main__":
    print("="*60)
    print("Measuring Inference Speed")
    print("="*60)
    
    # Load model
    model = SimpleDetector()
    model.load_state_dict(torch.load("detector.pth"))
    model.eval()
    
    # Measure on dummy data
    print("\n📊 Dummy Data (100 images):")
    fps_dummy, time_dummy = measure_fps(model, num_images=100)
    print(f"  FPS: {fps_dummy:.2f}")
    print(f"  Average Time: {time_dummy:.2f} ms/image")
    
    # Measure on real images
    print("\n📊 Real Images (Validation Set):")
    fps_real, time_real = measure_real_image_fps(model, "../../datasets/coco5/images/val", num_images=16)
    print(f"  FPS: {fps_real:.2f}")
    print(f"  Average Time: {time_real:.2f} ms/image")
    
    # Model size
    import os
    model_size = os.path.getsize("detector.pth") / (1024 * 1024)  # MB
    print(f"\n📦 Model Size: {model_size:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total Parameters: {total_params:,}")
    print(f"🔢 Trainable Parameters: {trainable_params:,}")
    
    print("="*60)
