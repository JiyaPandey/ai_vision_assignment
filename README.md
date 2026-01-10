# AI Vision - Object Detection Model

A computer vision project that detects everyday objects in real-time using a custom-trained deep learning model.

---

## 📖 Project Description

This is an **object detection and computer vision model** that identifies and localizes common objects in images and live video streams. The model uses transfer learning with a **ResNet18 backbone** pre-trained on ImageNet, fine-tuned for custom object detection tasks.

### What Does This Project Do?

- **Detects and localizes objects** in images and videos
- **Draws bounding boxes** around detected objects with class labels
- **Supports real-time detection** through webcam feed
- **Provides a complete demo** showcasing the model's capabilities

### Supported Classes

The model can detect the following **5 object classes**:

1. **iPad** - Tablet devices
2. **Backpack** - Bags and backpacks
3. **Hand** - Human hands
4. **Phone** - Mobile phones and smartphones
5. **Wallet** - Wallets and purses

---

## 🚀 How to Run the Project

### First-Time Setup

If you're running this project for the first time, follow these steps:

#### 1. Clone the Repository

```bash
git clone https://github.com/JiyaPandey/ai_vision_assignment.git
cd ai_vision_assignment/part1_object_detection
```

#### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

#### 3. Activate the Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

#### 4. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy tqdm
```

---

### Running the Model

#### Option 1: Run the Demo Script

The **demo script** showcases the complete working of the model by running detection on multiple test images.

**What the demo does:**
- Loads the trained model
- Processes random images from the validation dataset (or custom images from the `demoimg` folder)
- Displays detected objects with bounding boxes and class labels
- Cycles through multiple images automatically (press 'q' to skip)

**Run the demo:**

```bash
cd ~/ai_vision_assignment/part1_object_detection
source ../venv/bin/activate
python src/demo.py
```

#### Option 2: Run Inference on a Single Image

To test the model on a single random validation image:

```bash
cd ~/ai_vision_assignment/part1_object_detection
source ../venv/bin/activate
python src/infer_yolo.py
```

**Output:** The result will be saved as `output_yolo.jpg` in the current directory.

---

## 📹 Webcam Option - Real-Time Detection

You can **enable webcam mode** to detect objects in real-time using your computer's camera.

### How to Use Webcam Detection

1. Connect a webcam to your computer (or use the built-in camera)
2. Run the webcam detection script:

```bash
cd ~/ai_vision_assignment/part1_object_detection
source ../venv/bin/activate
python detect_webcam.py
```

3. **Show objects live** in front of the camera
4. The model will **detect objects in real-time** and display:
   - Bounding boxes around detected objects
   - Class labels with confidence scores
   - Live FPS (frames per second) counter

5. Press **'q'** to quit the webcam feed

### Important Note About Hand Detection

> **Note:** Hand detection may not always be accurate because the training dataset mostly contained American images with fair-skinned hands, which can affect generalization to other skin tones and hand variations. This is a known limitation of the current training data.

---

## 🎥 Demo Video

*A demo video showcasing the model's real-time detection capabilities will be added here soon.*

**Stay tuned!** The video will demonstrate:
- Object detection on various test images
- Live webcam detection
- Model performance and accuracy

---

## 📊 Model Performance

- **Architecture:** ResNet18 backbone with custom detection heads
- **Training Dataset:** Custom 5-class dataset (YOLO format)
- **Image Size:** 224x224 pixels
- **Transfer Learning:** Pre-trained on ImageNet

---

## 🛠️ Project Structure

```
part1_object_detection/
├── src/
│   ├── demo.py              # Demo script for multiple images
│   ├── model.py             # Model architecture (ImprovedDetector)
│   ├── train_yolo.py        # Training script
│   ├── infer_yolo.py        # Single image inference
│   └── yolo_dataset.py      # Dataset loader
├── detect_webcam.py         # Real-time webcam detection
├── detector_yolo_best.pth   # Trained model weights
└── README.md                # This file
```

---

## 📚 Technologies Used

- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision library
- **torchvision** - Pre-trained models and utilities
- **NumPy** - Numerical computations

---

## 👨‍💻 Author

**Jiya Pandey**  
GitHub: [@JiyaPandey](https://github.com/JiyaPandey)

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🌟 Acknowledgments

- **Pascal VOC Dataset** - For initial training concepts
- **ResNet Architecture** - Pre-trained backbone for transfer learning
- **PyTorch Community** - For excellent documentation and support

---

**Made with ❤️ for computer vision and AI**