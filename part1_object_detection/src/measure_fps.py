import time
import torch
from model import ImprovedDetector
import numpy as np

DEVICE = "cpu"  # Measure on CPU as requested

def measure_fps():
    model = ImprovedDetector().to(DEVICE)
    model.eval()
    
    # Dummy input
    input_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
            
    # Measure
    print("Measuring FPS...")
    num_frames = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_frames):
            _ = model(input_tensor)
            
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_frames / total_time
    avg_time_ms = (total_time / num_frames) * 1000
    
    print(f"Total time for {num_frames} frames: {total_time:.4f}s")
    print(f"FPS: {fps:.2f}")
    print(f"Average Inference Time: {avg_time_ms:.2f} ms")

if __name__ == "__main__":
    measure_fps()
