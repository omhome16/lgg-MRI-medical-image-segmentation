import torch
import torch.nn as nn
import torch.optim as optim

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss
    """

    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)

        # Combine losses
        combined_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return combined_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced segmentation
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        # Calculate focal weights
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_optimizer(model, optimizer_type='adam', lr=0.001, weight_decay=1e-5):
    """
    Create optimizer

    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'sgd', or 'adamw')
        lr: Learning rate
        weight_decay: Weight decay factor

    Returns:
        Optimizer
    """
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='step', step_size=7, gamma=0.1):
    """
    Create learning rate scheduler

    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('step', 'cosine', or 'plateau')
        step_size: Step size for StepLR
        gamma: Decay factor

    Returns:
        Scheduler
    """
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")