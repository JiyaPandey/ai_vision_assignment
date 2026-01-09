# AI Vision Assignment - Object Detection & Quality Inspection

This project implements two computer vision tasks: custom object detection from scratch and PCB quality inspection using YOLOv8.

---

## 📁 Project Structure

```
ai_vision_assignment/
├── part1_object_detection/      # Custom detector trained from scratch
├── part2_quality_inspection/    # YOLOv8-based PCB defect detection
├── datasets/                    # Pascal VOC, DeepPCB datasets
└── README.md                    # This file
```

---

## 🚀 Quick Start Commands

### Part 1: Object Detection from Scratch

**Train the model:**
```bash
cd ~/ai_vision_assignment/part1_object_detection
source ../venv/bin/activate
python src/train_voc.py
```

**Run inference:**
```bash
cd ~/ai_vision_assignment/part1_object_detection
source ../venv/bin/activate
python src/infer_voc.py
```

**Output:** `output_voc.jpg` (random validation image with detection)

---

### Part 2: PCB Quality Inspection

**Train YOLOv8 (if not already trained):**
```bash
cd ~/ai_vision_assignment
source venv/bin/activate
yolo detect train data=part2_quality_inspection/pcb.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8 patience=10
```

**Run inspection:**
```bash
cd ~/ai_vision_assignment
source venv/bin/activate
python part2_quality_inspection/pcb_inspect.py
```

**Output:** `part2_quality_inspection/results/output_images/inspected_*.jpg`

---

## 📊 Results Comparison

| Metric | Part 1 (From Scratch) | Part 2 (YOLOv8 Pretrained) |
|--------|----------------------|---------------------------|
| **Dataset** | Pascal VOC 2007 | DeepPCB |
| **Images** | 480 train / 120 val | 50 images |
| **Training Approach** | From scratch | Fine-tuning pretrained |
| **Validation Accuracy** | 29.2% | 99.2% mAP@0.5 |
| **IoU** | 0.278 | 0.99+ |
| **Training Time** | 50 minutes | 16 minutes |
| **Classes** | 5 (person, car, dog, cat, bicycle) | 3 (missing, misaligned, solder defects) |

---

## 📝 Key Insights

1. **Transfer Learning is Crucial**: Part 2 achieves 99.2% mAP with only 50 images by fine-tuning YOLOv8, while Part 1 achieves 29.2% accuracy with 480 images training from scratch.

2. **Pre-trained Weights Matter**: YOLOv8's COCO pre-training provides robust feature extractors that adapt quickly to new domains.

3. **Data Requirements**: Training from scratch requires significantly more data (10x+) to achieve comparable performance.

---

## 🛠️ Setup

**Install dependencies:**
```bash
cd ~/ai_vision_assignment
source venv/bin/activate
pip install torch torchvision opencv-python ultralytics
```

**Dataset locations:**
- Part 1: `~/ai_vision_assignment/datasets/VOCdevkit/VOC2007`
- Part 2: `~/ai_vision_assignment/part2_quality_inspection/data`

---

## 📚 Documentation

- [Part 1 README](part1_object_detection/README.md)
- [Part 1 Technical Report](part1_object_detection/TECHNICAL_REPORT.md)
- [Part 2 README](part2_quality_inspection/README.md)

---

## 🎯 Assignment Completion

✅ Part 1: Custom object detection from scratch (Pascal VOC 2007)  
✅ Part 2: Quality inspection with YOLOv8 (DeepPCB)  
✅ Training & evaluation scripts  
✅ Inference demos  
✅ Comprehensive documentation  
