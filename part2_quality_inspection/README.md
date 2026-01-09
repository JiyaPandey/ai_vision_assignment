# Part 2: Quality Inspection System

## Overview

**Product**: PCB (Printed Circuit Board) manufacturing quality control

**Defects Detected**:
- Missing components
- Misaligned components
- Solder defects

## Dataset

**Source**: DeepPCB dataset (50 PCB images with defects)

**Annotations**: Automatically converted from DeepPCB format to YOLO format using `convert_annotations.py`. Original 6 classes mapped to 3 custom defect categories.

## Model

**Architecture**: YOLOv8n (nano) - pretrained on COCO dataset

**Why Chosen**:
- Pre-trained weights accelerate training on small datasets
- State-of-the-art detection accuracy
- Fast inference speed (124ms per image on CPU)
- Purpose-built for real-time object detection

## Inspection Pipeline

**Detection**: YOLOv8 identifies defect bounding boxes

**Classification**: Each detection classified into one of 3 defect types

**Coordinates**: Center (x, y) coordinates calculated from bounding box

**Severity Logic**: 
- HIGH: defect area > 5% of image
- MEDIUM: defect area > 2% of image  
- LOW: defect area < 2% of image

## How to Run

```bash
python part2_quality_inspection/pcb_inspect.py
```

## Demo Output

**Output Location**: `part2_quality_inspection/results/output_images/inspected_*.jpg`

Images show bounding boxes with defect labels and confidence scores.

**Training Results**: `runs/detect/train3/`
- Best model: mAP@0.5 = 99.2%
- Per-class performance: 96.8-100% recall
