import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_loss = nn.MSELoss()
        # Class weights to handle imbalance: person=205, car=26, dog=8, bottle=17, chair=20
        class_weights = torch.tensor([1.0, 7.88, 25.63, 12.06, 10.25])
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, preds, targets):
        # preds: [B, 4 + C]
        # targets: [B, 5] → [cls, x, y, w, h]

        box_pred = preds[:, :4]
        cls_pred = preds[:, 4:]

        box_gt = targets[:, 1:]
        cls_gt = targets[:, 0].long()

        loss_box = self.box_loss(box_pred, box_gt)
        loss_cls = self.cls_loss(cls_pred, cls_gt)

        # Balance: emphasize bbox learning
        return 3.0 * loss_box + loss_cls
