"""
Prepares the dataset for segmentation by processing image and mask files.

To execute the script, use the following command:
    `python -m src.prepare_data_segmentation --input-folder <path_to_input_dir>`

Args:
    input_folder (str): Path to the directory containing the `train` and `test` subdirectories,
                        with `images` and `masks` subdirectories inside each of them.

Example:
    `python -m src.prepare_data_segmentation --input-directory ./data/cbis-ddsm-detec`
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


def crop_image_to_mask(image, mask, min_padding=50, max_padding=300):
    """Crops an image based on the mask's white area with random padding."""
    mask_array = np.array(mask)

    # Find white pixels
    y_indices, x_indices = np.where(mask_array > 0)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return image, mask

    # Get bounding box
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Apply padding
    paddings = [np.random.randint(min_padding, max_padding) for _ in range(4)]
    x_min = max(x_min - paddings[0], 0)
    x_max = min(x_max + paddings[1], image.width)
    y_min = max(y_min - paddings[2], 0)
    y_max = min(y_max + paddings[3], image.height)

    # Crop the image and mask
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

    return cropped_image, cropped_mask


def process_dataset(input_folder, output_folder, dataset_type):
    """Loads images, finds masks, crops them, and saves to output folders."""
    image_folder = os.path.join(input_folder, dataset_type, "images")
    mask_folder = os.path.join(input_folder, dataset_type, "masks")

    if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
        print(f"⚠️ Skipping {dataset_type} dataset (folders not found).")
        return 0

    output_image_folder = os.path.join(output_folder, dataset_type, "images")
    output_mask_folder = os.path.join(output_folder, dataset_type, "masks")

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    total_images = len(image_files)

    if total_images == 0:
        print(f"⚠️ No images found in {dataset_type} dataset.")
        return 0

    print(f"📂 Processing {dataset_type} dataset: {total_images} images found.")

    for index, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path)

        # Find corresponding masks
        mask_pattern = os.path.join(
            mask_folder, f"{os.path.splitext(img_file)[0]}_*.png"
        )
        mask_files = sorted(glob.glob(mask_pattern))
        masks = [Image.open(mask_path) for mask_path in mask_files]

        # Process each mask
        for i, mask in enumerate(masks):
            cropped_image, cropped_mask = crop_image_to_mask(image, mask)

            # Save cropped images and masks
            cropped_image.save(
                os.path.join(
                    output_image_folder, f"{os.path.splitext(img_file)[0]}_{i}.png"
                )
            )
            cropped_mask.save(
                os.path.join(
                    output_mask_folder, f"{os.path.splitext(img_file)[0]}_mask_{i}.png"
                )
            )

        # Print progress update
        print(
            f"✅ Processed {index}/{total_images} images in {dataset_type}.",
            end="\r",
            flush=True,
        )

    print(f"\n🎉 Finished processing {dataset_type}. {total_images} images processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images based on mask regions.")
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Path to the input data folder (e.g., '../data/cbis-ddsm')",
    )

    args = parser.parse_args()

    output_folder = f"{args.input_folder}-segme"

    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)

    for dataset in ["train", "test", "val"]:
        for type in ["calc", "mass"]:
            dt = os.path.join(dataset, type)
            process_dataset(args.input_folder, output_folder, dt)
