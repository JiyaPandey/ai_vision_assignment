import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class Coco5Dataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224):
        """
        root_dir: path to datasets/coco5
        split: 'train' or 'val'
        img_size: image resize size
        """
        self.img_dir = os.path.join(root_dir, "images", split)
        self.lbl_dir = os.path.join(root_dir, "labels", split)
        self.img_size = img_size

        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image path
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0  # normalize to 0–1

        # Convert to tensor (C, H, W)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # Label path
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(self.lbl_dir, label_name)

        boxes = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    boxes.append([cls, x, y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        return img, boxes
