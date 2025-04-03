import pytorch_lightning as pl
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets.unet import UNet
from torch.optim import AdamW


class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_path=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
        )

        self.dice_metric = DiceMetric(reduction="mean")
        self.dice_loss = DiceLoss(sigmoid=False, softmax=True)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        dice_metric = self.dice_metric(
            torch.argmax(outputs, dim=1).unsqueeze(0),
            torch.argmax(masks, dim=1).unsqueeze(0),
        )
        dice_metric = torch.mean(dice_metric)
        dice_loss = self.dice_loss(outputs, masks)
        ce_loss = self.ce_loss(outputs, masks)
        loss = dice_loss + ce_loss
        self.log("train_dice", dice_metric, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        with torch.no_grad():
            outputs = self(images)
        dice_metric = self.dice_metric(
            torch.argmax(outputs, dim=1).unsqueeze(0),
            torch.argmax(masks, dim=1).unsqueeze(0),
        )
        dice_metric = torch.mean(dice_metric)
        dice_loss = self.dice_loss(outputs, masks)
        ce_loss = self.ce_loss(outputs, masks)
        val_loss = dice_loss + ce_loss
        self.log("val_dice", dice_metric, prog_bar=True, on_epoch=True)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
