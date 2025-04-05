import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from src.data_modules.detection_data_module import CbisDdsmDataModuleDetection
from src.data_modules.segmentation_data_module import CbisDdsmDataModuleSegmentation
from src.models.detection_model import DetectionModel
from src.models.segmentation_model import SegmentationModel


def train_detection_model(filename, max_epochs=150):
    datamodule = CbisDdsmDataModuleDetection(
        root_dir="data/cbis-ddsm-detec",
        tumor_type="mass",
        batch_size=1,
        num_workers=1,
    )
    model = DetectionModel()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="models/",
        save_weights_only=True,
        filename=filename,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    wandb_logger = pl.loggers.WandbLogger(project=f"mammo-vision-{filename}")
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, datamodule=datamodule)


def train_segmentation_model(filename, max_epochs=150):
    datamodule = CbisDdsmDataModuleSegmentation(
        root_dir="data/cbis-ddsm-segme",
        tumor_type="mass",
        batch_size=8,
        num_workers=7,
    )
    model = SegmentationModel()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="models/",
        save_weights_only=False,
        filename=filename,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    wandb_logger = pl.loggers.WandbLogger(project=f"mammo-vision-{filename}")
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, datamodule=datamodule)


def train():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    wandb.login()
    train_detection_model("mass-detection")
    train_segmentation_model("mass-segmentation")
    wandb.finish()
