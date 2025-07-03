import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random


def explore_lgg_dataset(data_dir):
    """Explore LGG MRI Segmentation dataset"""
    # Get directories
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))]

    # Count images and masks
    total_images = 0
    images_with_tumor = 0

    for patient_dir in patient_dirs:
        files = os.listdir(patient_dir)
        images = [f for f in files if not f.endswith('_mask.tif')]
        masks = [f for f in files if f.endswith('_mask.tif')]

        total_images += len(images)

        # Check if masks contain tumors
        for mask_file in masks:
            mask_path = os.path.join(patient_dir, mask_file)
            mask = np.array(Image.open(mask_path))
            if mask.sum() > 0:
                images_with_tumor += 1

    print(f"Dataset Statistics:")
    print(f"- Total patients: {len(patient_dirs)}")
    print(f"- Total images: {total_images}")
    print(f"- Images with tumor: {images_with_tumor}")
    print(f"- Images without tumor: {total_images - images_with_tumor}")
    print(f"- Tumor prevalence: {images_with_tumor / total_images:.2%}")

    # Visualize a few examples
    visualize_random_samples(patient_dirs, num_samples=5)


def visualize_random_samples(patient_dirs, num_samples=5):
    """Visualize random samples from the dataset"""
    plt.figure(figsize=(15, num_samples * 5))

    for i in range(num_samples):
        # Select random patient
        patient_dir = random.choice(patient_dirs)

        # Get image and mask files
        files = os.listdir(patient_dir)
        images = [f for f in files if not f.endswith('_mask.tif')]

        # Select random image
        img_file = random.choice(images)
        mask_file = img_file.replace('.tif', '_mask.tif')

        # Check if mask exists
        if mask_file not in files:
            continue

        # Load image and mask
        img_path = os.path.join(patient_dir, img_file)
        mask_path = os.path.join(patient_dir, mask_file)

        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))

        # Display
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.title(f"Patient ID: {os.path.basename(patient_dir)}")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.title(f"Tumor Mask")
        plt.imshow(mask)
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.title(f"Overlay")
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='Reds')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('data_samples.png')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explore LGG MRI dataset")
    parser.add_argument('--data_dir', type=str, default='./data/lgg-mri-segmentation',
                        help='Path to dataset')
    args = parser.parse_args()

    explore_lgg_dataset(args.data_dir)