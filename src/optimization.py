import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold

from src import train_model


def apply_mixup(images, masks, alpha=0.2):
    """
    Apply Mixup data augmentation to images and masks

    Args:
        images: Batch of images
        masks: Batch of masks
        alpha: Alpha parameter for Beta distribution

    Returns:
        Mixed images and masks
    """
    batch_size = images.size(0)

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = torch.from_numpy(lam).float().to(images.device)
    lam = lam.view(-1, 1, 1, 1)

    # Permute batch indices
    index = torch.randperm(batch_size).to(images.device)

    # Mix images and masks
    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_masks = lam * masks + (1 - lam) * masks[index]

    return mixed_images, mixed_masks


def apply_cutmix(images, masks, alpha=1.0):
    """
    Apply CutMix data augmentation to images and masks

    Args:
        images: Batch of images
        masks: Batch of masks
        alpha: Alpha parameter for Beta distribution

    Returns:
        Mixed images and masks
    """
    batch_size, _, height, width = images.size()

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Permute batch indices
    index = torch.randperm(batch_size).to(images.device)

    # Calculate bounding box for mixing
    cut_width = int(width * lam ** 0.5)
    cut_height = int(height * lam ** 0.5)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_width // 2, 0, width)
    bby1 = np.clip(cy - cut_height // 2, 0, height)
    bbx2 = np.clip(cx + cut_width // 2, 0, width)
    bby2 = np.clip(cy + cut_height // 2, 0, height)

    # Apply CutMix
    mixed_images = images.clone()
    mixed_masks = masks.clone()

    # Cut and paste the patches
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
    mixed_masks[:, :, bby1:bby2, bbx1:bbx2] = masks[index, :, bby1:bby2, bbx1:bbx2]

    return mixed_images, mixed_masks


def k_fold_cross_validation(dataset, model_fn, criterion, optimizer_fn, scheduler_fn,
                            num_folds=5, num_epochs=25, batch_size=8, device='cuda'):
    """
    Perform k-fold cross-validation

    Args:
        dataset: Full dataset
        model_fn: Function to create model
        criterion: Loss function
        optimizer_fn: Function to create optimizer
        scheduler_fn: Function to create scheduler
        num_folds: Number of folds
        num_epochs: Number of epochs per fold
        batch_size: Batch size
        device: Device to use

    Returns:
        List of models, mean and std of validation dice scores
    """
    # Define KFold cross-validator
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Save validation scores for each fold
    fold_val_scores = []

    # Save models for each fold
    fold_models = []

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    for fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        print(f"Fold {fold + 1}/{num_folds}")
        print("-" * 30)

        # Sample elements randomly from a given list of indices
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        # Define data loaders for training and validation
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        # Create model, optimizer, and scheduler
        model = model_fn().to(device)
        optimizer = optimizer_fn(model)
        scheduler = scheduler_fn(optimizer)

        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device
        )

        # Save model and validation score
        fold_models.append(model)
        fold_val_scores.append(history['val_dice'][-1])

        print(f"Fold {fold + 1} - Validation Dice: {history['val_dice'][-1]:.4f}")

    # Calculate mean and standard deviation of validation scores
    mean_val_score = np.mean(fold_val_scores)
    std_val_score = np.std(fold_val_scores)

    print(f"K-fold Cross-validation - Mean Dice: {mean_val_score:.4f} ± {std_val_score:.4f}")

    return fold_models, mean_val_score, std_val_score


def create_ensemble(models, inputs, threshold=0.5):
    """
    Create ensemble prediction from multiple models

    Args:
        models: List of trained models
        inputs: Input images
        threshold: Threshold for binary segmentation

    Returns:
        Ensemble prediction
    """
    # Switch models to evaluation mode
    for model in models:
        model.eval()

    # Initialize predictions
    preds_sum = torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3))).to(inputs.device)

    # Get predictions from each model
    with torch.no_grad():
        for model in models:
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds_sum += probs

    # Average predictions
    preds_avg = preds_sum / len(models)

    # Apply threshold
    preds = (preds_avg > threshold).float()

    return preds


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge Loss for binary segmentation
    Based on Lovasz-Softmax: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, pred, target):
        return lovasz_hinge(pred, target)


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
    logits: [B, H, W] or [B, 1, H, W] of unnormalized probabilities
    labels: [B, H, W] or [B, 1, H, W] where each value is 0 or 1
    per_image: compute the loss per image instead of per batch
    ignore: void class id
    """
    if per_image:
        loss = torch.mean(lovasz_hinge_flat(*flatten_binary_scores(log.squeeze(1), lab.squeeze(1), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits.squeeze(1), labels.squeeze(1), ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss (flat)
    logits: [P] unnormalized scores
    labels: [P] binary labels (0 or 1)
    """
    if len(labels) == 0:
        # Only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation
    https://arxiv.org/abs/1706.05721
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate true positives, false positives, and false negatives
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()

        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Return Tversky loss
        return 1 - tversky