import os
import re

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.transform_utils import RandomFlip, RandomZoom


def list_image_paths(root_dir):
    img_paths = []
    for filename in os.listdir(root_dir):
        img_path = os.path.join(root_dir, filename)
        img_paths.append(img_path)
    return img_paths


class CbisDdsmDatasetDetection(Dataset):
    """
    CBIS-DDSM dataset.
    This Dataset class can be used for both detection and segmentation tasks.
    """

    def __init__(self, root_dir, transform=None):
        img_dir = os.path.join(root_dir, "images")
        msk_dir = os.path.join(root_dir, "masks")

        assert os.path.exists(root_dir)
        assert os.path.exists(img_dir)
        assert os.path.exists(msk_dir)

        self.img_paths = list_image_paths(img_dir)
        self.msk_paths = list_image_paths(msk_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        image = TF.to_tensor(image).float()

        _, base_name = os.path.split(img_path)
        base_name, _ = os.path.splitext(base_name)

        mask_pattern = re.compile(f"{re.escape(base_name)}_(calc|mass)_\\d+\\.png$")
        masks_path = [
            path
            for path in self.msk_paths
            if mask_pattern.match(os.path.basename(path))
        ]
        boxes = []
        labels = []

        for msk_path in masks_path:
            _, filename = os.path.split(msk_path)
            if filename.startswith(base_name):
                mask = Image.open(msk_path).convert("L")
                mask = np.array(mask)

                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()

                    padding = 50
                    x_min = np.clip(x_min - padding, 0, mask.shape[1])
                    x_max = np.clip(x_max + padding, 0, mask.shape[1])
                    y_min = np.clip(y_min - padding, 0, mask.shape[0])
                    y_max = np.clip(y_max + padding, 0, mask.shape[0])
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]  # Small dummy box
            labels = [0]  # Background class (0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        sample = (image, target)

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_fn(batch):
    return tuple(zip(*batch))


class CbisDdsmDataModuleDetection(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CBIS-DDSM dataset.
    This DataModule class can be used for both detection and segmentation tasks.
    """

    def __init__(self, root_dir, tumor_type="mass", batch_size=5, num_workers=0):
        super().__init__()

        assert tumor_type in ["calc", "mass"]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = CbisDdsmDatasetDetection(
            root_dir=os.path.join(root_dir, "train", tumor_type),
            transform=transforms.Compose(
                [RandomFlip(0.5, 0.5), RandomZoom((1, 3), 0.5)]
            ),
        )

        self.val_dataset = CbisDdsmDatasetDetection(
            os.path.join(root_dir, "val", tumor_type)
        )

        self.test_dataset = CbisDdsmDatasetDetection(
            os.path.join(root_dir, "test", tumor_type)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
