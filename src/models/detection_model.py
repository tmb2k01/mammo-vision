import pytorch_lightning as pl
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DetectionModel(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3, weight_path=None):
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn(
            weights=(
                FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if weight_path is None else None
            ),
        )

        # Freeze backbone (ResNet50)
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Replace ROI head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if weight_path:
            self.load_weights(weight_path)

        self.lr = lr

    def load_weights(self, weight_path):
        """Load model weights from a given file path."""
        checkpoint = torch.load(weight_path, map_location=self.device)

        state_dict = {
            key.replace("model.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }

        self.model.load_state_dict(state_dict)
        print(f"Loaded weights from {weight_path}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        train_loss = sum(loss for loss in loss_dict.values())
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(images),
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        with torch.no_grad():  # Ensure gradients are not calculated
            loss_dict = self.model(images, targets)
        val_loss = sum(loss for loss in loss_dict.values())
        self.model.eval()
        self.log(
            "val_loss", val_loss, prog_bar=True, on_epoch=True, batch_size=len(images)
        )
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
