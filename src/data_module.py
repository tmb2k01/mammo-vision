import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def list_image_paths(root_dir):
    img_paths = []
    for filename in os.listdir(root_dir):
        img_path = os.path.join(root_dir, filename)
        img_paths.append(img_path)
    return img_paths


class CbisDdsmDataset(Dataset):
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

        masks = []
        _, base_name = os.path.split(img_path)
        base_name, _ = os.path.splitext(base_name)

        for msk_path in self.msk_paths:
            _, filename = os.path.split(msk_path)
            if filename.startswith(base_name):
                mask = Image.open(msk_path)
                mask = np.array(mask)
                masks.append(mask)

        # Ensure that a black mask of the same size as the image is created
        image_width, image_height = image.size
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for mask in masks:
            height, width = mask.shape
            assert width == image_width and height == image_height
            combined_mask = np.logical_or(combined_mask, mask)

        combined_mask = combined_mask.astype(np.uint8)
        combined_mask = TF.to_pil_image(combined_mask)

        sample = (image, combined_mask)

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip:
    def __init__(self, ph, pv):
        assert isinstance(ph, float)
        assert isinstance(pv, float)

        self.ph = ph
        self.pv = pv

    def __call__(self, sample):
        image, mask = sample

        if np.random.random() < self.ph:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if np.random.random() < self.pv:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return (image, mask)


def zoom(image, zoom_factor):
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    top = (height - new_height) // 2
    left = (width - new_width) // 2

    image = TF.crop(image, top, left, new_height, new_width)
    image = TF.resize(image, (height, width))

    return image


class RandomZoom:
    def __init__(self, zoom_factor, p):
        assert isinstance(zoom_factor, (int, float))
        assert isinstance(p, float)
        self.zoom_factor = zoom_factor
        self.p = p

    def __call__(self, sample):
        image, mask = sample

        if np.random.random() < self.p:
            image = zoom(image, self.zoom_factor)
            mask = zoom(mask, self.zoom_factor)

        return (image, mask)


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.resize = transforms.Resize(output_size)

    def __call__(self, sample):
        image, mask = sample

        image = self.resize(image)
        mask = self.resize(mask)

        return (image, mask)


class ToTensor:
    def __call__(self, sample):
        image, mask = sample

        image = TF.to_tensor(image).to(torch.uint8)
        mask = TF.to_tensor(mask).to(torch.uint8)

        return (image, mask)


class CbisDdsmDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CBIS-DDSM dataset.
    This DataModule class can be used for both detection and segmentation tasks.
    """

    def __init__(self, root_dir, tumor_type, batch_size, num_workers):
        super().__init__()

        assert tumor_type in ["calc", "mass"]

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = CbisDdsmDataset(
            root_dir=os.path.join(root_dir, "train", tumor_type),
            transform=transforms.Compose(
                [
                    RandomFlip(0.5, 0.5),
                    RandomZoom(1.5, 0.5),
                    Resize((416, 416)),
                    ToTensor(),
                ]
            ),
        )

        self.val_dataset = CbisDdsmDataset(
            os.path.join(root_dir, "val", tumor_type),
            transform=transforms.Compose(
                [
                    Resize((416, 416)),
                    ToTensor(),
                ]
            ),
        )

        self.test_dataset = CbisDdsmDataset(
            os.path.join(root_dir, "test", tumor_type),
            transform=transforms.Compose(
                [
                    Resize((416, 416)),
                    ToTensor(),
                ]
            ),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
