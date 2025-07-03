import os
import numpy as np
from PIL import Image
import argparse


def preprocess_dataset(data_dir, output_dir):
    """Preprocess LGG MRI Segmentation dataset"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    # Get directories
    patient_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))]

    image_count = 0

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        files = os.listdir(patient_dir)

        # Get images and corresponding masks
        images = [f for f in files if not f.endswith('_mask.tif')]

        for img_file in images:
            mask_file = img_file.replace('.tif', '_mask.tif')

            # Check if mask exists
            if mask_file not in files:
                continue

            # Load image and mask
            img_path = os.path.join(patient_dir, img_file)
            mask_path = os.path.join(patient_dir, mask_file)

            image = Image.open(img_path)
            mask = Image.open(mask_path)

            # Resize to standard size
            image = image.resize((256, 256))
            mask = mask.resize((256, 256))

            # Save with standardized filename
            image_filename = f"{patient_id}_{image_count:04d}.png"
            mask_filename = f"{patient_id}_{image_count:04d}_mask.png"

            image.save(os.path.join(output_dir, 'images', image_filename))
            mask.save(os.path.join(output_dir, 'masks', mask_filename))

            image_count += 1

    print(f"Preprocessed {image_count} images and saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LGG MRI dataset")
    parser.add_argument('--data_dir', type=str, default='./data/lgg-mri-segmentation',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./data/preprocessed',
                        help='Output directory')

    args = parser.parse_args()

    preprocess_dataset(args.data_dir, args.output_dir)