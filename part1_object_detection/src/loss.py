import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_iou_loss(pred_boxes, target_boxes):
    """
    Compute IoU loss for bounding boxes
    pred_boxes: [B, 4] (x_center, y_center, width, height) - normalized
    target_boxes: [B, 4] (x_center, y_center, width, height) - normalized
    """
    # Convert to x1, y1, x2, y2
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    
    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    # IoU loss = 1 - IoU
    return 1.0 - iou.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('alpha_buffer', alpha if alpha is not None else torch.tensor([1.0]))
    
    def forward(self, inputs, targets):
        alpha = self.alpha_buffer if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ImprovedDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Class weights based on your data distribution
        class_weights = torch.tensor([1.5, 1.5, 1.0, 1.2, 1.0])  
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)

    def forward(self, preds, targets):
        # preds: [B, 4 + C]
        # targets: [B, 5] → [cls, x, y, w, h]

        box_pred = preds[:, :4]
        cls_pred = preds[:, 4:]

        box_gt = targets[:, 1:]
        cls_gt = targets[:, 0].long()

        # IoU loss for bounding boxes
        loss_box = compute_iou_loss(box_pred, box_gt)
        
        # Focal loss for classification
        loss_cls = self.focal_loss(cls_pred, cls_gt)

        # Much higher weight on bounding box to improve localization
        # Classification is already good (98%), focus on bbox
        return 10.0 * loss_box + 1.0 * loss_cls

