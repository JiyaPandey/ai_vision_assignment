import cv2
import os
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Our trained model
IMAGE_PATH = "part2_quality_inspection/data/images"
OUTPUT_DIR = "part2_quality_inspection/results/output_images"

CLASSES = {
    0: "missing_component",
    1: "misaligned_component",
    2: "solder_defect"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# Pick one image for demo
img_name = os.listdir(IMAGE_PATH)[0]
img_path = os.path.join(IMAGE_PATH, img_name)

image = cv2.imread(img_path)
h, w, _ = image.shape

# ---------------- RUN INFERENCE ----------------
results = model(img_path, conf=0.3)

print(f"\nInspecting image: {img_name}\n")

for r in results:
    boxes = r.boxes
    if boxes is None:
        continue

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Center coordinates
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Severity logic (area-based)
        box_area = (x2 - x1) * (y2 - y1)
        img_area = w * h
        area_ratio = box_area / img_area

        if area_ratio > 0.05:
            severity = "HIGH"
        elif area_ratio > 0.02:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        defect_name = CLASSES.get(cls_id, "unknown")

        # Print structured output
        print(f"Defect: {defect_name}")
        print(f"Confidence: {conf:.2f}")
        print(f"Center (x, y): ({cx}, {cy})")
        print(f"Severity: {severity}\n")

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{defect_name} ({conf:.2f})"
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ---------------- SAVE OUTPUT ----------------
out_path = os.path.join(OUTPUT_DIR, f"inspected_{img_name}")
cv2.imwrite(out_path, image)

print(f"Inspection result saved to: {out_path}")
