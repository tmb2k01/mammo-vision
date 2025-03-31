import os

import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.transform_utils import RandomFlip, Resize


def list_image_paths(root_dir):
    img_paths = []
    for filename in os.listdir(root_dir):
        img_path = os.path.join(root_dir, filename)
        img_paths.append(img_path)
    return img_paths


class CbisDdsmDatasetSegmentation(Dataset):
    """
    CBIS-DDSM dataset.
    This Dataset class can be used for both detection and segmentation tasks.
    """

    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.msk_dir = os.path.join(root_dir, "masks")

        assert os.path.exists(root_dir)
        assert os.path.exists(self.img_dir)
        assert os.path.exists(self.msk_dir)

        self.img_paths = list_image_paths(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = os.path.join(
            self.msk_dir,
            os.path.basename(img_path).replace(".png", "_mask.png"),
        )
        image = Image.open(img_path)
        image = TF.to_tensor(image).float()
        image = (image - image.mean()) / (image.std() + 1e-8)

        mask = Image.open(mask_path)
        mask = TF.to_tensor(mask).long()
        mask = one_hot(mask.squeeze(0), num_classes=2).permute(2, 0, 1).float()

        sample = (image, mask)

        if self.transform:
            sample = self.transform(sample)

        return sample


class CbisDdsmDataModuleSegmentation(pl.LightningDataModule):
    def __init__(self, root_dir, tumor_type="mass", batch_size=5, num_workers=0):
        super().__init__()

        assert tumor_type in ["calc", "mass"]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = CbisDdsmDatasetSegmentation(
            root_dir=os.path.join(root_dir, "train", tumor_type),
            transform=transforms.Compose([RandomFlip(0.5, 0.5), Resize((128, 128))]),
        )

        self.val_dataset = CbisDdsmDatasetSegmentation(
            os.path.join(root_dir, "val", tumor_type),
            transform=transforms.Compose([Resize((128, 128))]),
        )

        self.test_dataset = CbisDdsmDatasetSegmentation(
            os.path.join(root_dir, "test", tumor_type),
            transform=transforms.Compose([Resize((128, 128))]),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
