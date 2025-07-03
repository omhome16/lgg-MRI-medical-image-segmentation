import sys
sys.path.append('./src')
from dataset import prepare_lgg_dataset
from model import get_model
from loss import BCEDiceLoss, get_optimizer, get_scheduler
from train import train_model, evaluate_model, visualize_results, plot_training_history
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime


# Import local modules - ensure these are in the same directory
# from dataset import prepare_lgg_dataset
# from model import get_model
# from loss import BCEDiceLoss, get_optimizer, get_scheduler
# from train import train_model, evaluate_model, visualize_results, plot_training_history

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare dataset
    print("Preparing dataset...")
    train_dataset, val_dataset, test_dataset = prepare_lgg_dataset(args.data_dir)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(
        f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")

    # Create model
    print(f"Creating {args.model_type} model...")
    model = get_model(model_type=args.model_type, n_channels=3, n_classes=1)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create loss function
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, optimizer_type=args.optimizer, lr=args.learning_rate,
                              weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_type=args.scheduler, step_size=args.step_size, gamma=args.gamma)

    # Train model
    print(f"Training model for {args.epochs} epochs...")
    start_time = time.time()

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save model
    model_path = os.path.join(run_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training history
    plot_training_history(history)
    plt.savefig(os.path.join(run_dir, 'training_history.png'))

    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_dice, vis_imgs, vis_masks, vis_preds = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Save results
    results = {
        'test_loss': test_loss,
        'test_dice': test_dice,
        'training_time': training_time,
        'parameters': sum(p.numel() for p in model.parameters()),
        'hyperparameters': vars(args)
    }

    # Save results to file
    with open(os.path.join(run_dir, 'results.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    # Visualize results
    visualize_results(vis_imgs, vis_masks, vis_preds)
    plt.savefig(os.path.join(run_dir, 'segmentation_results.png'))

    print(f"All results saved to {run_dir}")
    print(f"Test Dice Score: {test_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LGG MRI Segmentation")

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./lgg-mri-segmentation', help='Path to dataset')
    parser.add_argument('--results_dir', type=str, default='./results', help='Path to save results')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='attention_unet', choices=['unet', 'attention_unet'],
                        help='Model type')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--bce_weight', type=float, default=0.5, help='BCE weight in combined loss')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau'],
                        help='Scheduler type')
    parser.add_argument('--step_size', type=int, default=7, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)