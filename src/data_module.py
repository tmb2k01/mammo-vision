import os

import numpy as np
import pytorch_lightning as pl
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
                masks.append(mask)

        sample = {"image": image, "masks": masks}

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
        image, masks = sample["image"], sample["masks"]

        if np.random.random() < self.ph:
            image = TF.hflip(image)
            masks = [TF.hflip(m) for m in masks]

        if np.random.random() < self.pv:
            image = TF.vflip(image)
            masks = [TF.vflip(m) for m in masks]

        return {"image": image, "masks": masks}


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
        image, masks = sample["image"], sample["masks"]

        if np.random.random() < self.p:
            image = zoom(image, self.zoom_factor)
            masks = [zoom(m, self.zoom_factor) for m in masks]

        return {"image": image, "masks": masks}


class Resize:
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.resize = transforms.Resize(output_size)

    def __call__(self, sample):
        image, masks = sample["image"], sample["masks"]

        image = self.resize(image)
        masks = [self.resize(m) for m in masks]

        return {"image": image, "masks": masks}


class ToTensor:
    def __call__(self, sample):
        image, masks = sample["image"], sample["masks"]

        image = TF.to_tensor(image)
        masks = [TF.to_tensor(m) for m in masks]

        return {"image": image, "masks": masks}


class CbisDdsmDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CBIS-DDSM dataset.
    This DataModule class can be used for both detection and segmentation tasks.
    """

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = CbisDdsmDataset(
            root_dir=os.path.join(root_dir, "train"),
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
            os.path.join(root_dir, "val"),
            transform=transforms.Compose(
                Resize((416, 416)),
                ToTensor(),
            ),
        )

        self.test_dataset = CbisDdsmDataset(
            os.path.join(root_dir, "test"),
            transform=transforms.Compose(
                Resize((416, 416)),
                ToTensor(),
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
