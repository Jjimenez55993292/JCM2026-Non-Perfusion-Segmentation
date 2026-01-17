# metrics.py

import torch

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """
    pred, target shape: (B,1,H,W) with values 0/1
    """
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean().item()

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """
    IoU = intersection / union
    """
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def sigmoid_to_binary(logits: torch.Tensor, thresh=0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > thresh).float()
