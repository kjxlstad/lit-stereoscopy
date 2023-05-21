from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from models import UNet
from data.datamodule import SceneFlowDataModule

torch.set_float32_matmul_precision("medium")

t = Trainer(accelerator="cuda", max_epochs=100, logger=TensorBoardLogger(""))

datamodule = SceneFlowDataModule(
    root_dir=Path("/media/jo/Flash/datasets/sceneflow"),
    res=(224, 224),
    grayscale=True,
    batch_size=48,
    num_workers=12,
    pin_memory=True,
)

model = UNet(in_channels=2, out_channels=1, init_features=32)

t.fit(model, datamodule=datamodule)
t.test(model, datamodule=datamodule)
