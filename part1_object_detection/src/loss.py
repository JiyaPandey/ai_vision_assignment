import torch
import torch.nn as nn

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
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ImprovedDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Higher weights for underrepresented classes (Ipad, backpack)
        # Based on your results: Ipad=0%, backpack=0%, hand=80%, phone=43%, wallet=38%
        class_weights = torch.tensor([3.0, 3.0, 1.0, 1.5, 1.5])  # Boost Ipad and backpack
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
        
        # Focal loss for classification (handles imbalance better)
        loss_cls = self.focal_loss(cls_pred, cls_gt)

        # Balance: emphasize both equally with focal loss
        return 3.0 * loss_box + 2.0 * loss_cls

    """Original detection loss - kept for compatibility"""
    def __init__(self):
        super().__init__()
        self.box_loss = nn.MSELoss()
        class_weights = torch.tensor([1.0, 7.88, 25.63, 12.06, 10.25])
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, preds, targets):
        box_pred = preds[:, :4]
        cls_pred = preds[:, 4:]

        box_gt = targets[:, 1:]
        cls_gt = targets[:, 0].long()

        loss_box = self.box_loss(box_pred, box_gt)
        loss_cls = self.cls_loss(cls_pred, cls_gt)

        return 3.0 * loss_box + loss_cls
