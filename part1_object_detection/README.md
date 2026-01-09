# Part 1: Custom Object Detector from Scratch

## Dataset
- **Source**: COCO-style 5-class subset
- **Classes**: person, car, dog, bottle, chair
- **Format**: YOLO format labels (normalized coordinates)
- **Split**: 80 total images
  - Training: 64 images
  - Validation: 16 images

## Model Architecture
- **Type**: Custom CNN built from scratch (no pretrained weights)
- **Architecture**: SimpleDetector
  - Convolutional backbone with 3 conv layers
  - Adaptive average pooling
  - Fully connected layers for regression + classification
- **Output**: 9 values per image
  - 4 for bounding box (x_center, y_center, width, height)
  - 5 for class probabilities

## Training
- **Device**: CPU
- **Epochs**: 20
- **Batch Size**: 8
- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: Combined MSE (bbox) + CrossEntropy (class)

### Training Loss Trend
```
Epoch 1/20  | Loss: 11.4634
Epoch 5/20  | Loss: 5.5620
Epoch 10/20 | Loss: 0.9509
Epoch 15/20 | Loss: 0.5669
Epoch 20/20 | Loss: 0.2955
```
**Observation**: Loss decreases steadily from ~11.5 to ~0.3, showing good convergence.

## Model Metrics
- **Model Size**: 50 MB
- **CPU Inference Speed**: ~50-100ms per image (CPU-dependent)
- **Evaluation**: IoU@0.5 / Qualitative results on validation set
  - Model successfully detects and classifies objects in test images
  - Bounding boxes are reasonably accurate for simple scenes

## Output
- Trained model: `detector.pth`
- Sample inference: `output.jpg` (image with bounding box + label)

## How to Run

### Training
```bash
cd part1_object_detection/src
python train.py
```

### Inference
```bash
cd part1_object_detection/src
python infer.py
```

## Files
- `src/model.py` - CNN architecture
- `src/loss.py` - Combined loss function
- `src/train.py` - Training loop
- `src/infer.py` - Inference demo
- `src/dataset.py` - Custom dataset loader
