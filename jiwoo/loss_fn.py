import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, focal_loss = FocalLoss()):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = focal_loss
        
        
    def forward(self, inputs, targets,smooth = 1e-5):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs*targets).sum(dim=(2,3))
        union = inputs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice = 2.0 * (intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice
        dice_loss = torch.mean(dice_loss)
        
        focal_loss = self.focal_loss(inputs,targets)
        
        loss = dice_loss+focal_loss
        return dice_loss, focal_loss, loss

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold):
    SMOOTH = 1e-9

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    outputs = torch.where(outputs>threshold,1,0).to(torch.uint8)
    labels = labels.squeeze(1).to(torch.uint8)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch