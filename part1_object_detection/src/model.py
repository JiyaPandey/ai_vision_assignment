import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImprovedDetector(nn.Module):
    """
    Transfer Learning based detector using pretrained ResNet18 backbone
    - Pretrained on ImageNet for better feature extraction
    - Fine-tuned for object detection
    - Much better accuracy with limited data
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        # Use pretrained ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # ResNet18 outputs 512 channels at 7x7 spatial size for 224x224 input
        # Add detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4 + num_classes)  # 4 bbox coords + class logits
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


# Alias for backward compatibility if needed, or just use ImprovedDetector
SimpleDetector = ImprovedDetector
x = self.detection_head(x)
        return x