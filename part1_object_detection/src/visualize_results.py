import torch
import cv2
import os
import numpy as np
from model import ImprovedDetector
from dataset import Coco5Dataset
import imageio

CLASSES = ["person", "car", "dog", "bottle", "chair"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_results(model_path="detector_improved_best.pth", output_gif="detection_demo.gif"):
    # Load model
    model = ImprovedDetector().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model {model_path} not found, using random weights for demo")
    
    model.eval()
    
    # Load validation dataset
    val_dataset = Coco5Dataset("../../datasets/coco5", split="val")
    
    frames = []
    
    print("Generating visualization...")
    
    for idx in range(len(val_dataset)):
        img_tensor, boxes = val_dataset[idx]
        
        # Original image for visualization
        img_vis = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        h, w, _ = img_vis.shape
        
        # Inference
        with torch.no_grad():
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_tensor)[0].cpu()
        
        # Parse prediction
        pred_box = output[:4]
        pred_cls_logits = output[4:]
        pred_cls = pred_cls_logits.argmax().item()
        pred_conf = torch.softmax(pred_cls_logits, dim=0).max().item()
        
        # Draw Ground Truth (Green)
        if len(boxes) > 0:
            gt_cls = int(boxes[0, 0].item())
            gt_box = boxes[0, 1:]
            
            x1 = int((gt_box[0] - gt_box[2]/2) * w)
            y1 = int((gt_box[1] - gt_box[3]/2) * h)
            x2 = int((gt_box[0] + gt_box[2]/2) * w)
            y2 = int((gt_box[1] + gt_box[3]/2) * h)
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"GT: {CLASSES[gt_cls]}"
            cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Draw Prediction (Red)
        x1 = int((pred_box[0] - pred_box[2]/2) * w)
        y1 = int((pred_box[1] - pred_box[3]/2) * h)
        x2 = int((pred_box[0] + pred_box[2]/2) * w)
        y2 = int((pred_box[1] + pred_box[3]/2) * h)
        
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Pred: {CLASSES[pred_cls]} ({pred_conf:.2f})"
        cv2.putText(img_vis, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add to frames
        frames.append(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        
        # Save individual image
        if idx < 5:
            cv2.imwrite(f"vis_result_{idx}.jpg", img_vis)
            
    # Save GIF
    imageio.mimsave(output_gif, frames, fps=2)
    print(f"Saved visualization to {output_gif}")

if __name__ == "__main__":
    visualize_results()
