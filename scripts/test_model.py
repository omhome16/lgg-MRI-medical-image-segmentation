import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import argparse

# Import local modules - adjust the path as needed
import sys

sys.path.append('./src')
from dataset import prepare_lgg_dataset
from model import get_model
from loss import BCEDiceLoss
from train import evaluate_model, visualize_results


def test_saved_model(model_path, data_dir, model_type='attention_unet', batch_size=8):
    """Test a saved model on the test dataset"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset
    print("Preparing dataset...")
    _, _, test_dataset = prepare_lgg_dataset(data_dir)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create model
    print(f"Creating {model_type} model...")
    model = get_model(model_type=model_type, n_channels=3, n_classes=1)

    # Load model weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Create loss function
    criterion = BCEDiceLoss()

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_dice, vis_imgs, vis_masks, vis_preds = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")

    # Visualize results
    print("Visualizing results...")
    visualize_results(vis_imgs, vis_masks, vis_preds)

    return test_loss, test_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LGG MRI Segmentation model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_dir', type=str, default='./data/lgg-mri-segmentation',
                        help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='attention_unet',
                        choices=['unet', 'attention_unet'], help='Model type')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    test_saved_model(args.model_path, args.data_dir, args.model_type, args.batch_size)