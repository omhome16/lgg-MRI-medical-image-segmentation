import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class LGGMRIDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        """
        Args:
            img_paths (list): List of paths to MRI images
            mask_paths (list): List of paths to segmentation masks
            transform (callable, optional): Optional transform to be applied on images
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert to grayscale

        # Apply transformations if available
        if self.transform:
            image = self.transform(image)
            # For masks, we want to ensure they remain binary after transformation
            mask = self.transform(mask)
            # Threshold masks to binary (0 or 1)
            mask = (mask > 0.5).float()

        return image, mask


def prepare_lgg_dataset(base_path):
    """
    Prepare the LGG MRI dataset paths
    Args:
        base_path (str): Base path to the dataset
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Find all patient folders
    patient_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if
                    os.path.isdir(os.path.join(base_path, d))]

    all_img_paths = []
    all_mask_paths = []

    # Collect all image and mask paths
    for patient_dir in patient_dirs:
        # Get all files in the patient directory
        files = os.listdir(patient_dir)

        # Separate images and masks
        imgs = [f for f in files if not f.endswith('_mask.tif')]
        masks = [f for f in files if f.endswith('_mask.tif')]

        # Match images with their masks
        for img_file in imgs:
            mask_file = img_file.replace('.tif', '_mask.tif')
            if mask_file in masks:
                all_img_paths.append(os.path.join(patient_dir, img_file))
                all_mask_paths.append(os.path.join(patient_dir, mask_file))

    # Split into train, validation, and test sets (70%, 15%, 15%)
    train_img, temp_img, train_mask, temp_mask = train_test_split(
        all_img_paths, all_mask_paths, test_size=0.3, random_state=42)

    val_img, test_img, val_mask, test_mask = train_test_split(
        temp_img, temp_mask, test_size=0.5, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = LGGMRIDataset(train_img, train_mask, transform)
    val_dataset = LGGMRIDataset(val_img, val_mask, transform)
    test_dataset = LGGMRIDataset(test_img, test_mask, transform)

    return train_dataset, val_dataset, test_dataset

