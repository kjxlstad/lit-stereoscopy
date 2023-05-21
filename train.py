from pathlib import Path

import torch
from lightning import Trainer

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from models import UNet
from data.datamodule import SceneFlowDataModule

torch.set_float32_matmul_precision("medium")

datamodule = SceneFlowDataModule(
    root_dir=Path("/media/jo/Flash/datasets/sceneflow"),
    res=(224, 224),
    grayscale=True,
    batch_size=32,
    num_workers=10,
    pin_memory=True,
)

model = UNet(in_channels=2, out_channels=1, init_features=32)

trainer = Trainer(
    accelerator="cuda",
    max_epochs=100,
    logger=TensorBoardLogger("lightning_logs", name=model.__class__.__name__),
    callbacks=[ModelCheckpoint(filename="best", monitor="val/epe", mode="min")],
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
