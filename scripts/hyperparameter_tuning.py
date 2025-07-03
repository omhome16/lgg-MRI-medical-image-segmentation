import sys

sys.path.append('./src')
from dataset import prepare_lgg_dataset
from model import get_model
from loss import BCEDiceLoss, get_optimizer, get_scheduler
from train import train_model, evaluate_model

import torch
from torch.utils.data import DataLoader
import itertools
import os
import json
import time


def hyperparameter_tuning(data_dir, results_dir):
    """Perform hyperparameter tuning"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset, val_dataset, _ = prepare_lgg_dataset(data_dir)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Define hyperparameter grid
    hyperparams = {
        'model_type': ['unet', 'attention_unet'],
        'learning_rate': [0.01, 0.001, 0.0001],
        'optimizer': ['adam', 'adamw'],
        'scheduler': ['step', 'cosine']
    }

    # Generate all combinations
    param_combos = list(itertools.product(*hyperparams.values()))
    param_names = list(hyperparams.keys())

    results = []

    for idx, combo in enumerate(param_combos):
        # Create parameter dictionary
        params = dict(zip(param_names, combo))
        print(f"Testing combination {idx + 1}/{len(param_combos)}: {params}")

        # Create model
        model = get_model(model_type=params['model_type'], n_channels=3, n_classes=1)

        # Create criterion, optimizer, and scheduler
        criterion = BCEDiceLoss()
        optimizer = get_optimizer(model, optimizer_type=params['optimizer'], lr=params['learning_rate'])
        scheduler = get_scheduler(optimizer, scheduler_type=params['scheduler'])

        # Train model
        start_time = time.time()
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=5,  # Use fewer epochs for hyperparameter tuning
            device=device
        )
        training_time = time.time() - start_time

        # Save results
        result = {
            'params': params,
            'val_dice': history['val_dice'][-1],
            'training_time': training_time
        }
        results.append(result)

        print(f"Validation Dice: {result['val_dice']:.4f}, Training time: {result['training_time']:.2f}s")

    # Sort results by validation Dice
    results.sort(key=lambda x: x['val_dice'], reverse=True)

    # Save results to file
    with open(os.path.join(results_dir, 'hyperparameter_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Best hyperparameters: {results[0]['params']}")
    print(f"Best validation Dice: {results[0]['val_dice']:.4f}")

    return results[0]['params']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument('--data_dir', type=str, default='./data/lgg-mri-segmentation',
                        help='Path to dataset')
    parser.add_argument('--results_dir', type=str, default='./results/hyperparameter_tuning',
                        help='Results directory')

    args = parser.parse_args()

    hyperparameter_tuning(args.data_dir, args.results_dir)