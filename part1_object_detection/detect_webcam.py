"""
Real-time object detection using webcam
Press 'q' to quit
"""
import cv2
import torch
import numpy as np
from model import ImprovedDetector

CLASSES = ['Ipad', 'backpack', 'hand', 'phone', 'wallet']
COLORS = [
    (255, 0, 0),    # Blue for Ipad
    (0, 255, 0),    # Green for backpack
    (0, 0, 255),    # Red for hand
    (255, 255, 0),  # Cyan for phone
    (255, 0, 255)   # Magenta for wallet
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    # Resize to 224x224
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Convert to tensor
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

def draw_detection(frame, bbox, class_id, confidence=None):
    """Draw bounding box and label on frame"""
    x1, y1, x2, y2 = bbox
    color = COLORS[class_id]
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label = CLASSES[class_id]
    if confidence is not None:
        label = f"{label} {confidence:.2f}"
    
    # Background for text
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    print("Loading model...")
    model = ImprovedDetector(num_classes=5, pretrained=False).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("detector_yolo_best.pth", map_location=DEVICE))
        print("✅ Loaded best model")
    except:
        print("⚠️  Best model not found, trying final model...")
        model.load_state_dict(torch.load("detector_yolo_final.pth", map_location=DEVICE))
        print("✅ Loaded final model")
    
    model.eval()
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit")
    print("=" * 50)
    
    fps_counter = 0
    fps_timer = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        h, w, _ = frame.shape
        
        # Preprocess and run inference
        with torch.no_grad():
            inp = preprocess_frame(frame)
            out = model(inp)[0].cpu()
        
        # Parse output
        x, y, bw, bh = out[:4]
        cls_logits = out[4:]
        cls = cls_logits.argmax().item()
        confidence = torch.softmax(cls_logits, dim=0)[cls].item()
        
        # Convert normalized coordinates to pixels
        cx, cy = int(x * w), int(y * h)
        box_w, box_h = int(bw * w), int(bh * h)
        
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w, cx + box_w // 2)
        y2 = min(h, cy + box_h // 2)
        
        # Draw detection
        draw_detection(frame, (x1, y1, x2, y2), cls, confidence)
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps = 30 / ((cv2.getTickCount() - fps_timer) / cv2.getTickFrequency())
            fps_timer = cv2.getTickCount()
            fps_counter = 0
        else:
            fps = 0
        
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Object Detection - Press Q to quit', frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:  # q, Q, or ESC
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam detection stopped")

if __name__ == "__main__":
    main()
