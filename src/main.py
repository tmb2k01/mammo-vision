import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from data_modules.detection_data_module import (
    CbisDdsmDataModuleDetection as DetectionDataModule,
)
from models.detection_model import DetectionModel


def _get_detection_module() -> DetectionDataModule:
    data_module = DetectionDataModule(
        root_dir="data/cbis-ddsm-detec", tumor_type="mass", batch_size=5, num_workers=4
    )
    return data_module


def _train_model(
    model: LightningModule,
    data_module: LightningDataModule,
    filename: str,
    max_epochs: int = 50,
) -> None:
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
    trainer.fit(model, datamodule=data_module)
    wandb.finish()


def _train() -> None:
    wandb.login()
    mass_data_module = _get_detection_module()

    mass_data_module = DetectionModel()

    # Train detection model
    _train_model(mass_data_module, mass_data_module, "mass-detection")


def _main() -> None:
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # train_mode = int(os.environ["TRAIN_MODE"])
    # print(f"TRAIN_MODE={train_mode}")
    # if train_mode == 1:
    print("Starting up training...")
    _train()
    # else:
    #     print("Loading model and serving requests...")
    #     # launch()


if __name__ == "__main__":
    _main()
