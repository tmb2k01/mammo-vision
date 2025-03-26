import pytorch_lightning as pl
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNETR
from torch.optim import AdamW


class SegmentationModel(pl.LightningModule):
    def __init__(self, in_channels=1, num_classes=2, lr=1e-3, weight_path=None):
        super().__init__()

        self.model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(256, 256),
            spatial_dims=2,
        )

        self.loss_fn = DiceLoss(softmax=True)

        if weight_path:
            self.load_weights(weight_path)

        self.lr = lr

    def load_weights(self, weight_path):
        """Load model weights from a given file path."""
        checkpoint = torch.load(
            weight_path, weights_only=True, map_location=self.device
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Loaded weights from {weight_path}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        val_loss = self.loss_fn(outputs, masks)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
