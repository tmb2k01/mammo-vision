"""
Prepares the dataset by processing image and mask files.

To execute the script, use the following command:
    `python -m src.prepare_data --input-directory <path_to_input_dir>`

Args:
    input_directory (str): Path to the directory containing the `train` and `test` subdirectories,
                            with `images` and `masks` subdirectories inside each of them.

Example:
    `python -m src.prepare_data --input-directory ./data/cbis-ddsm`
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

    # Collect patient IDs from filenames
    patients = {f.split("_")[1] for f in os.listdir(src_img_dir)}

    # Randomly select 10% of patients for validation
    random.seed(42)
    patients = sorted(patients)
    val_patients = random.sample(patients, int(0.1 * len(patients)))

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


def prepare_dir(in_dir, subdir, out_dir):
    in_img_dir = os.path.join(in_dir, subdir, "images")
    in_msk_dir = os.path.join(in_dir, subdir, "masks")
    out_img_dir = os.path.join(out_dir, subdir, "images")
    out_msk_dir = os.path.join(out_dir, subdir, "masks")

    assert os.path.exists(in_img_dir)
    assert os.path.exists(in_msk_dir)

    os.makedirs(out_img_dir)
    os.makedirs(out_msk_dir)

    in_img_names = os.listdir(in_img_dir)
    img_cnt = len(in_img_names)
    print(f"INFO: Processing '{in_img_dir}': {img_cnt} images found")

    # Apply CLAHE to each image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i, img_name in enumerate(in_img_names, start=1):
        in_img_path = os.path.join(in_img_dir, img_name)

        image = Image.open(in_img_path)
        image = clahe.apply(np.array(image, dtype=np.uint8))
        image = Image.fromarray(image)

        out_img_path = os.path.join(out_img_dir, img_name)
        image.save(out_img_path)

        print(f"INFO: Processed {i}/{img_cnt} images", end="\r", flush=True)

    # Copy mask files into a directory the CbisDdsmDataset will find
    print(f"INFO: Copying masks from '{in_msk_dir}' to '{out_msk_dir}'")
    shutil.copytree(in_msk_dir, out_msk_dir)

    print(f"INFO: Finished processing '{in_dir}'")


def prepare_dataset(in_dir):
    train_dir = os.path.join(in_dir, "train")
    test_dir = os.path.join(in_dir, "test")

    assert os.path.exists(in_dir)
    assert os.path.exists(train_dir)
    assert os.path.exists(test_dir)

    out_dir = f"{in_dir}-prepared"
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
