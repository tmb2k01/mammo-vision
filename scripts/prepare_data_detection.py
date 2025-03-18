"""
Prepares the dataset for detection by processing image and mask files.

To execute the script, use the following command:
    `python -m src.prepare_data_detection --input-directory <path_to_input_dir>`

Args:
    input_directory (str): Path to the directory containing the `train` and `test` subdirectories,
                            with `images` and `masks` subdirectories inside each of them.

Example:
    `python -m src.prepare_data_detection --input-directory ./data/cbis-ddsm`
"""

import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image


def move_file(src_dir, dst_dir, filename):
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    shutil.move(src, dst)


def move_masks(src_msk_dir, dst_msk_dir, filename):
    base_name = os.path.splitext(filename)[0]
    msk_pattern = os.path.join(src_msk_dir, f"{base_name}_*.png")
    for msk_src in glob.glob(msk_pattern):
        base_name = os.path.basename(msk_src)
        msk_dst = os.path.join(dst_msk_dir, base_name)
        shutil.move(msk_src, msk_dst)


def create_val_dir(in_dir, subdir):
    src_img_dir = os.path.join(in_dir, subdir, "images")
    src_msk_dir = os.path.join(in_dir, subdir, "masks")
    dst_img_dir = os.path.join(in_dir, "val", "images")
    dst_msk_dir = os.path.join(in_dir, "val", "masks")

    assert os.path.exists(src_img_dir)
    assert os.path.exists(src_msk_dir)

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_msk_dir, exist_ok=True)

    # Collect patients by density class
    density_classes = {1: [], 2: [], 3: [], 4: []}

    for filename in os.listdir(src_img_dir):
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split("_")
        density = int(parts[-1])
        patient_id = parts[1]
        density_classes[density].append(patient_id)

    # Move 10% of patients from each density class to validation
    random.seed(42)
    val_patients = []

    for density in density_classes:
        density_patients = sorted(set(density_classes[density]))
        num_val_patients = int(0.1 * len(density_patients))
        val_class_patients = random.sample(density_patients, num_val_patients)
        val_patients.extend(val_class_patients)
        print(f"INFO: Moving {num_val_patients} patients from density class {density}")

    # Move files for validation set
    for filename in os.listdir(src_img_dir):
        patient_id = filename.split("_")[1]
        if patient_id in val_patients:
            move_file(src_img_dir, dst_img_dir, filename)
            move_masks(src_msk_dir, dst_msk_dir, filename)

    print(f"INFO: Moved {len(val_patients)} patients to validation")


def list_image_paths(root_dir):
    img_paths = []
    for filename in os.listdir(root_dir):
        img_path = os.path.join(root_dir, filename)
        img_paths.append(img_path)
    return img_paths


def has_mask_of_type(img_name, tumor_type, msk_dir):
    assert tumor_type in ["calc", "mass"]
    for mask_name in os.listdir(msk_dir):
        mask_base_name = os.path.splitext(mask_name)[0]
        mask_prefix = mask_base_name.rsplit("_", 2)[0]
        if f"_{tumor_type}_" in mask_base_name and mask_prefix in img_name:
            return True
    return False


def prepare_dir(in_dir, subdir, out_dir):
    in_img_dir = os.path.join(in_dir, subdir, "images")
    in_msk_dir = os.path.join(in_dir, subdir, "masks")
    out_calc_img_dir = os.path.join(out_dir, subdir, "calc", "images")
    out_calc_msk_dir = os.path.join(out_dir, subdir, "calc", "masks")
    out_mass_img_dir = os.path.join(out_dir, subdir, "mass", "images")
    out_mass_msk_dir = os.path.join(out_dir, subdir, "mass", "masks")

    assert os.path.exists(in_img_dir)
    assert os.path.exists(in_msk_dir)

    os.makedirs(out_calc_img_dir, exist_ok=True)
    os.makedirs(out_calc_msk_dir, exist_ok=True)
    os.makedirs(out_mass_img_dir, exist_ok=True)
    os.makedirs(out_mass_msk_dir, exist_ok=True)

    in_img_names = os.listdir(in_img_dir)
    img_cnt = len(in_img_names)
    print(f"INFO: Processing '{in_img_dir}': {img_cnt} images found")

    successful_image_basenames = []

    # Apply CLAHE to each image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i, img_name in enumerate(in_img_names, start=1):
        in_img_path = os.path.join(in_img_dir, img_name)

        image = Image.open(in_img_path)

        try:
            image = clahe.apply(np.array(image, dtype=np.uint8))
        except:
            print(f"WARN: Couldn't apply CLAHE to '{in_img_path}'")
            print(f"WARN: Skipping image '{in_img_path}' due to processing failure")
            continue

        image = Image.fromarray(image)

        if has_mask_of_type(img_name, "calc", in_msk_dir):
            out_calc_img_path = os.path.join(out_calc_img_dir, img_name)
            image.save(out_calc_img_path)

        if has_mask_of_type(img_name, "mass", in_msk_dir):
            out_mass_img_path = os.path.join(out_mass_img_dir, img_name)
            image.save(out_mass_img_path)

        successful_image_basenames.append(os.path.splitext(img_name)[0])

        print(f"INFO: Processed {i}/{img_cnt} images", end="\r", flush=True)

    # Copy mask files into a directory the CbisDdsmDataset will find
    print(f"INFO: Copying masks from '{in_msk_dir}' to '{out_calc_msk_dir}' and '{out_mass_msk_dir}'")

    for mask_name in os.listdir(in_msk_dir):
        mask_base_name = os.path.splitext(mask_name)[0]
        mask_prefix = mask_base_name.rsplit("_", 2)[0]
        if mask_prefix in successful_image_basenames:
            in_msk_path = os.path.join(in_msk_dir, mask_name)
            # Determine the appropriate directory for each mask
            if "_calc_" in mask_name:
                out_msk_path = os.path.join(out_calc_msk_dir, mask_name)
            else:
                out_msk_path = os.path.join(out_mass_msk_dir, mask_name)
            shutil.copy(in_msk_path, out_msk_path)

    print(f"INFO: Finished processing '{os.path.join(in_dir, subdir)}'")


def prepare_dataset(in_dir):
    train_dir = os.path.join(in_dir, "train")
    test_dir = os.path.join(in_dir, "test")

    assert os.path.exists(in_dir)
    assert os.path.exists(train_dir)
    assert os.path.exists(test_dir)

    out_dir = f"{in_dir}-detec"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"WARN: Removed '{out_dir}'")

    # Use a small portion of the training data for validation
    create_val_dir(in_dir, "train")

    prepare_dir(in_dir, "train", out_dir)
    prepare_dir(in_dir, "val", out_dir)
    prepare_dir(in_dir, "test", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", "-i", type=str, required=True)
    args = parser.parse_args()
    prepare_dataset(args.input_directory)
