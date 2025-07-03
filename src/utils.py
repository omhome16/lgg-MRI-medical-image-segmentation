import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_summary(model, file_path):
    """Save model summary to text file"""
    with open(file_path, 'w') as f:
        total_params = count_parameters(model)
        f.write(f"Model has {total_params:,} trainable parameters\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}: {param.numel():,}\n")