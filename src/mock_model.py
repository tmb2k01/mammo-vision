"""
Run this script to validate that the LightningDataModule loads data correctly.

Usage:
    python -m src.mock_model
"""

import pytorch_lightning as pl
import torch

import src.data_module as dm


class MockModel(pl.LightningModule):
    """
    A minimal LightningModule used for validating the functionality of the CbisDdsmDataModule.
    """

    def forward(self, images):
        assert isinstance(images, torch.Tensor)
        assert images.ndimension() == 4

        return images

    def training_step(self, batch):
        images, masks = batch

        assert isinstance(images, torch.Tensor)
        assert isinstance(masks, torch.Tensor)

        assert images.ndimension() == 4
        assert masks.ndimension() == 4

        return torch.tensor([0.0], requires_grad=True)

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    data_module = dm.CbisDdsmDataModule(
        root_dir="data/cbis-ddsm-detec",
        tumor_type="calc",
        batch_size=5,
        num_workers=7,
    )
    model = MockModel()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, datamodule=data_module)
