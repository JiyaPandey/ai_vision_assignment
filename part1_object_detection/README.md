# Part 1: Custom Object Detection from Scratch

## Overview
Complete object detection pipeline trained **from scratch** (no pre-trained weights) on Pascal VOC 2007 dataset.

## Dataset
- **Source**: Pascal VOC 2007
- **Classes**: person, car, dog, cat, bicycle (5 classes)
- **Format**: XML annotations converted to normalized bounding boxes
- **Split**: 600 total images
  - Training: 480 images (80%)
  - Validation: 120 images (20%)

## Model Architecture
- **Type**: Custom CNN built from scratch (no pretrained weights)
- **Name**: ImprovedDetector
- **Architecture**: 
  - Convolutional backbone: 3 conv layers (16→32→64 channels)
  - Adaptive pooling
  - Fully connected head with dropout (0.3)
  - Direct bounding box regression + classification
- **Output**: 9 values per image
  - 4 for bounding box (x_center, y_center, width, height) - normalized
  - 5 for class probabilities
- **Parameters**: ~3 million

## Training Details
- **Device**: CPU
- **Epochs**: 100 (early stopping at epoch 50)
- **Best Epoch**: 35
- **Batch Size**: 16
- **Optimizer**: Adam (lr=1e-3) with cosine annealing
- **Loss Function**: 
  - MSE (bounding box regression)
  - Weighted CrossEntropy (classification)
  - Combined: `3.0 × MSE + WeightedCrossEntropy`
- **Data Augmentation**:
  - Random brightness (0.7-1.3×)
  - Random horizontal flip (50%)
  - Resize to 224×224
- **Early Stopping**: Patience = 15 epochs

## Performance Results
- **Best Validation IoU**: 0.278
- **Best Validation Accuracy**: 29.2%
- **Training Time**: ~50 minutes (CPU only)

## How to Run

### Train the Model
```bash
cd part1_object_detection
source ../venv/bin/activate
python src/train_voc.py
```

### Run Inference
```bash
cd part1_object_detection
source ../venv/bin/activate
python src/infer_voc.py
```

Output saved as `output_voc.jpg` with bounding box and class label.

## Project Structure
```
part1_object_detection/
├── src/
│   ├── voc_dataset.py      # Pascal VOC dataset loader
│   ├── model.py            # ImprovedDetector architecture
│   ├── loss.py             # Combined loss function
│   ├── train_voc.py        # Training script
│   └── infer_voc.py        # Inference script
├── create_voc_split.py     # Creates train/val split
├── detector_voc_best.pth   # Best trained model weights
└── README.md
```

## Key Challenges

1. **Training from Scratch**: Without pre-trained weights, the model requires more data and longer training to achieve reasonable performance.

2. **Small Dataset Effect**: Even with 480 training images, performance is limited compared to models using transfer learning.

3. **Class Imbalance**: Pascal VOC has varying instance counts per class, requiring weighted loss.

4. **Bounding Box Localization**: Single-object regression is challenging without anchor boxes or region proposals.

## Comparison with Part 2

| Metric | Part 1 (From Scratch) | Part 2 (YOLOv8 Pretrained) |
|--------|----------------------|---------------------------|
| Dataset Size | 480 images | 50 images |
| Training Approach | From scratch | Fine-tuning pretrained |
| Validation Accuracy | 29.2% | 99.2% mAP@0.5 |
| IoU | 0.278 | 0.99+ |
| Training Time | 50 minutes | 16 minutes |

**Key Lesson**: Transfer learning (Part 2) dramatically outperforms training from scratch (Part 1) when dataset size is limited. Pre-trained weights provide better feature representations, enabling high accuracy even with minimal data.
