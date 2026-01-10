import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImprovedDetector(nn.Module):
    """
    Transfer Learning based detector using pretrained ResNet18 backbone
    - Pretrained on ImageNet for better feature extraction
    - Separate heads for classification and bounding box regression
    - Fine-tuned for object detection
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        # Use pretrained ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Shared feature processing
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        # Separate head for bounding box regression
        self.bbox_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # x, y, w, h
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
        
        # Separate head for classification
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.shared_head(x)
        
        # Separate predictions
        bbox = self.bbox_head(x)
        cls = self.cls_head(x)
        
        # Concatenate: [bbox (4), classification (num_classes)]
        return torch.cat([bbox, cls], dim=1)


# Alias for backward compatibility if needed, or just use ImprovedDetector
SimpleDetector = ImprovedDetector