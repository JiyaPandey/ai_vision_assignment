import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224, augment=False):
        """
        YOLO format dataset loader
        img_dir: path to images folder
        label_dir: path to labels folder (txt files)
        img_size: image resize size
        augment: whether to apply data augmentation
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        
        # 5 classes for the new dataset
        self.class_names = ['Ipad', 'backpack', 'hand', 'phone', 'wallet']
        self.num_classes = len(self.class_names)
        
        # Get all image files
        self.samples = []
        for img_name in os.listdir(img_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Check if corresponding label file exists
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    self.samples.append(img_name)
        
        # Normalize mean/std for ImageNet pretrained models
        self.normalize_mean = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])
        
        print(f"Loaded {len(self.samples)} images with labels")

    def _parse_label(self, label_path):
        """
        Parse YOLO format label file
        Format: class_id x_center y_center width height (all normalized)
        Returns first object as [class_id, x_center, y_center, width, height]
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                # Return dummy if empty
                return torch.tensor([0, 0.5, 0.5, 0.1, 0.1], dtype=torch.float32)
            
            # Parse first object
            parts = lines[0].strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            return torch.tensor([class_id, x_center, y_center, width, height], dtype=torch.float32)
        except:
            # Return dummy on error
            return torch.tensor([0, 0.5, 0.5, 0.1, 0.1], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation (more aggressive)
        if self.augment:
            # Random brightness and contrast
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.6, 1.4)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # Random contrast
            if np.random.rand() > 0.5:
                contrast = np.random.uniform(0.8, 1.2)
                img = np.clip(128 + contrast * (img - 128), 0, 255).astype(np.uint8)
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            
            # Random rotation (larger angle)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
            
            # Random blur
            if np.random.rand() > 0.7:
                ksize = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize with ImageNet statistics (for pretrained ResNet)
        img = img / 255.0
        img = (img - self.normalize_mean) / self.normalize_std
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        # Parse label
        target = self._parse_label(label_path)
        
        return img, target
