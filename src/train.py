import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Training function

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs
        device: Device to use

    Returns:
        Trained model and history of metrics
    """
    model = model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }

    best_val_dice = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Train phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        batch_count = 0

        for inputs, masks in train_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Calculate loss
            loss = criterion(probs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            train_dice += dice_coefficient(preds, masks).item()
            batch_count += 1

        # Learning rate adjustment
        scheduler.step()

        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_dice = train_dice / batch_count

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        batch_count = 0

        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                # Calculate loss
                loss = criterion(probs, masks)

                # Statistics
                val_loss += loss.item() * inputs.size(0)
                val_dice += dice_coefficient(preds, masks).item()
                batch_count += 1

        # Calculate epoch statistics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_dice = val_dice / batch_count

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_dice'].append(epoch_train_dice)
        history['val_dice'].append(epoch_val_dice)

        # Print epoch results
        print(f'Train Loss: {epoch_train_loss:.4f} Dice: {epoch_train_dice:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Dice: {epoch_val_dice:.4f}')

        # Save best model
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with Dice: {best_val_dice:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Calculate Dice coefficient

    Args:
        pred: Predicted binary segmentation
        target: Ground truth binary segmentation
        epsilon: Small constant to avoid division by zero

    Returns:
        Dice coefficient
    """
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Calculate Dice coefficient
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice


def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Evaluate model on test dataset

    Args:
        model: Trained model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use

    Returns:
        Test loss and Dice coefficient
    """
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_dice = 0.0
    batch_count = 0

    # Store predictions and ground truth for visualization
    all_imgs = []
    all_masks = []
    all_preds = []

    with torch.no_grad():
        for inputs, masks in test_loader:
            # Only store a few samples for visualization
            if len(all_imgs) < 5:
                all_imgs.append(inputs.cpu())
                all_masks.append(masks.cpu())

            inputs = inputs.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Store predictions for visualization
            if len(all_preds) < 5:
                all_preds.append(preds.cpu())

            # Calculate loss
            loss = criterion(probs, masks)

            # Statistics
            test_loss += loss.item() * inputs.size(0)
            test_dice += dice_coefficient(preds, masks).item()
            batch_count += 1

    # Calculate statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_dice = test_dice / batch_count

    print(f'Test Loss: {test_loss:.4f} Dice: {test_dice:.4f}')

    # Concatenate samples for visualization
    vis_imgs = torch.cat(all_imgs)
    vis_masks = torch.cat(all_masks)
    vis_preds = torch.cat(all_preds)

    return test_loss, test_dice, vis_imgs, vis_masks, vis_preds


def visualize_results(images, masks, predictions, num_samples=5):
    """
    Visualize segmentation results

    Args:
        images: Original images
        masks: Ground truth masks
        predictions: Predicted masks
        num_samples: Number of samples to visualize
    """
    plt.figure(figsize=(15, num_samples * 5))

    for i in range(min(num_samples, len(images))):
        # Display original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"Original Image {i + 1}")
        plt.imshow(images[i*2].permute(1, 2, 0))
        plt.axis('off')

        # Display ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"Ground Truth Mask {i + 1}")
        plt.imshow(masks[i*2].squeeze(), cmap='gray')
        plt.axis('off')

        # Display predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title(f"Predicted Mask {i + 1}")
        plt.imshow(predictions[i*2].squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_results1.png')
    plt.show()

    plt.figure(figsize=(15, num_samples * 5))

    for i in range(min(num_samples, len(images))):
        # Display original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"Original Image {i + 1}")
        plt.imshow(images[i*5].permute(1, 2, 0))
        plt.axis('off')

        # Display ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"Ground Truth Mask {i + 1}")
        plt.imshow(masks[i*5].squeeze(), cmap='gray')
        plt.axis('off')

        # Display predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title(f"Predicted Mask {i + 1}")
        plt.imshow(predictions[i*5].squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_results2.png')
    plt.show()

    plt.figure(figsize=(15, num_samples * 5))

    for i in range(min(num_samples, len(images))):
        # Display original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"Original Image {i + 1}")
        plt.imshow(images[i*7].permute(1, 2, 0))
        plt.axis('off')

        # Display ground truth mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"Ground Truth Mask {i + 1}")
        plt.imshow(masks[i*7].squeeze(), cmap='gray')
        plt.axis('off')

        # Display predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title(f"Predicted Mask {i + 1}")
        plt.imshow(predictions[i*7].squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_results3.png')
    plt.show()


def plot_training_history(history):
    """
    Plot training history

    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Dice coefficient
    plt.subplot(1, 2, 2)
    plt.title('Dice Coefficient')
    plt.plot(history['train_dice'], label='Train')
    plt.plot(history['val_dice'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()