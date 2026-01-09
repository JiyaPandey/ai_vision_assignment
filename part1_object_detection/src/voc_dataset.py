import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np


class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224, augment=False):
        """
        root_dir: path to VOCdevkit/VOC2007
        split: 'train' or 'val'
        img_size: image resize size
        augment: whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        self.img_size = img_size
        self.augment = augment
        
        # 5 classes we care about
        self.class_names = ["person", "car", "dog", "cat", "bicycle"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load image list
        split_file = os.path.join(root_dir, "ImageSets/Main", f"{split}.txt")
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
        
        # Filter images that contain our 5 classes
        self.samples = []
        for img_id in image_ids:
            ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
            if self._contains_target_classes(ann_path):
                self.samples.append(img_id)
        
        print(f"Loaded {len(self.samples)} images for {split} split with 5 classes")

    def _contains_target_classes(self, ann_path):
        """Check if annotation contains any of our target classes"""
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in self.class_names:
                    return True
            return False
        except:
            return False

    def _parse_annotation(self, ann_path):
        """Parse XML annotation and return first object of target class"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Find first object of target class
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_names:
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to normalized center format (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            class_id = self.class_to_idx[class_name]
            
            return torch.tensor([class_id, x_center, y_center, width, height], dtype=torch.float32)
        
        # Return dummy if no valid object found (shouldn't happen due to filtering)
        return torch.tensor([0, 0.5, 0.5, 0.1, 0.1], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        if self.augment:
            # Random brightness
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
        
        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize and convert to tensor
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        # Parse annotation
        target = self._parse_annotation(ann_path)
        
        return img, target
