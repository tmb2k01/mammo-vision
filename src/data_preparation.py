"""
1. Make sure the input directory contains subdirectories named `images` and `masks`.
2. Use the following command to run the script:

    python prepare_data.py --input-directory <path_to_input_directory> --target-size <target_image_size>

    - `<path_to_input_directory>`: Path to the directory containing `images` and `masks`.
    - `<target_image_size>`: The target size to which images will be resized (e.g., 224).

    Example:
    python -m src.prepare_data --input-directory ./data/cbis-ddsm/train --target-size 224
"""

import argparse
import glob
import os
import shutil
import sys
from typing import Dict, Tuple

import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


def zoom_image(image: Image.Image, zoom_factor: float = 1.5) -> Image.Image:
    """
    Zooms into the center of the image by a given factor, then resizes it back to the original size.

    Args:
        image (Image.Image): The input image.
        zoom_factor (float): The zoom factor (default is 1.5).

    Returns:
        Image.Image: The zoomed and resized image.
    """
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Calculate crop box (centered)
    top = (height - new_height) // 2
    left = (width - new_width) // 2

    cropped_image = TF.crop(image, top, left, new_height, new_width)
    resized_image = TF.resize(cropped_image, (height, width))

    return resized_image


def apply_image_transformations(
    image: Image.Image, target_size: Tuple[int, int]
) -> Dict[str, Image.Image]:
    """
    Applies CLAHE, flipping, and zoom transformations to an image and resizes them to the target size.

    Args:
        image (Image.Image): The input image.
        target_size (Tuple[int, int]): The desired output size of the images.

    Returns:
        Dict[str, Image.Image]: Dictionary containing transformed images.
    """
    transformations = {}

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(np.array(image, dtype=np.uint8))
    img_clahe = Image.fromarray(img_clahe)
    transformations["clahe"] = TF.resize(img_clahe, target_size)

    # Flip horizontally
    img_flip = TF.hflip(img_clahe)
    transformations["clahe_flip"] = TF.resize(img_flip, target_size)

    # Zoom the image
    img_zoom = zoom_image(img_clahe)
    transformations["clahe_zoom"] = TF.resize(img_zoom, target_size)

    # Flip and zoom the image
    img_flip_zoom = zoom_image(img_flip)
    transformations["clahe_flip_zoom"] = TF.resize(img_flip_zoom, target_size)

    return transformations


def prepare_data(input_directory: str, output_directory: str, target_size: int) -> None:
    """
    Prepares images and masks for training by applying transformations and saving them to the output directory.

    Args:
        input_directory (str): The directory containing the input images and masks.
        output_directory (str): The directory where the transformed images and masks will be saved.
        target_size (int): The target size for the transformed images.
    """
    images_directory = os.path.join(input_directory, "images")
    masks_directory = os.path.join(input_directory, "masks")

    # Check if directories exist
    if not os.path.exists(images_directory):
        print(f"⛔ Images directory '{input_directory}' does not exist.")
        sys.exit(1)

    if not os.path.exists(masks_directory):
        print(f"⛔ Masks directory '{input_directory}' does not exist.")
        sys.exit(1)

    # Create output directories
    output_images_directory = os.path.join(output_directory, "images")
    output_masks_directory = os.path.join(output_directory, "masks")

    os.makedirs(output_images_directory, exist_ok=True)
    os.makedirs(output_masks_directory, exist_ok=True)

    # Get list of image files
    images = sorted([f for f in os.listdir(images_directory) if f.endswith(".png")])
    image_count = len(images)

    if image_count == 0:
        print(f"⛔ No images found in '{images_directory}'.")
        sys.exit(1)

    print(f"📂 Processing data in '{input_directory}': {image_count} images found.")

    target_size = (target_size, target_size)

    # Process each image
    for index, img_file in enumerate(images, start=1):
        img_path = os.path.join(images_directory, img_file)
        image = Image.open(img_path)

        # Find corresponding mask files
        mask_pattern = f"{os.path.splitext(img_file)[0]}_*.png"
        mask_files = sorted(glob.glob(os.path.join(masks_directory, mask_pattern)))
        masks = [Image.open(mask_path) for mask_path in mask_files]

        # Save transformed images
        for name, transformed_image in apply_image_transformations(
            image, target_size
        ).items():
            transformed_image_path = os.path.join(
                output_images_directory, f"{os.path.splitext(img_file)[0]}_{name}.png"
            )
            transformed_image.save(transformed_image_path)

        # Save transformed masks
        for i, mask in enumerate(masks):
            for name, transformed_mask in apply_image_transformations(
                mask, target_size
            ).items():
                transformed_mask_path = os.path.join(
                    output_masks_directory,
                    f"{os.path.splitext(img_file)[0]}_mask_{i}_{name}.png",
                )
                transformed_mask.save(transformed_mask_path)

        print(f"✅ Processed {index}/{image_count} images.", end="\r", flush=True)

    print(f"\n🎉 Finished processing '{input_directory}'.")


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Prepare images for training")
    parser.add_argument(
        "--input-directory",
        "-i",
        type=str,
        required=True,
        help="Input directory with images and masks",
    )
    parser.add_argument(
        "--target-size",
        "-s",
        type=int,
        required=True,
        help="Target size for images",
    )
    args = parser.parse_args()

    output_directory = f"{args.input_directory}-prepared"

    # Check if input directory exists
    if not os.path.exists(args.input_directory):
        print(f"⛔ Input directory '{args.input_directory}' does not exist.")
        sys.exit(1)

    # Remove existing output directory if present
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        print(f"⚠️ Removed '{output_directory}' directory.")

    # Prepare data
    prepare_data(args.input_directory, output_directory, args.target_size)
