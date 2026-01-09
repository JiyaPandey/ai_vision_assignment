import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        # preds: [B, 4 + C]
        # targets: [B, 5] → [cls, x, y, w, h]

        box_pred = preds[:, :4]
        cls_pred = preds[:, 4:]

        box_gt = targets[:, 1:]
        cls_gt = targets[:, 0].long()

        loss_box = self.box_loss(box_pred, box_gt)
        loss_cls = self.cls_loss(cls_pred, cls_gt)

        return loss_box + loss_cls
