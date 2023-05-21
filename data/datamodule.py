from typing import get_args
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize, resize

from lightning import LightningDataModule
from lightning.pytorch.trainer.trainer import TrainerFn
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from data import data_paths, DataSplit
from data.dataset import SceneFlowDataset, ChainedDataset, Batch


class SceneFlowDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: Path,
        res: tuple[int, int],
        grayscale: bool = False,
        batch_size: int = 1,
        num_workers: int = 1,
        pin_memory: bool = True,
    ) -> None:
        super(SceneFlowDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.root_dir = root_dir
        self.res = res
        self.grayscale = grayscale

    def prepare_data(self) -> None:
        for split in get_args(DataSplit):
            data_paths(self.root_dir, split)

    def setup(self, stage: str) -> None:
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING):
            self.train_dataset = SceneFlowDataset(self.root_dir, "train", self.grayscale)
            self.validation_dataset = SceneFlowDataset(self.root_dir, "val", self.grayscale)

        if stage in TrainerFn.TESTING:
            # TODO: Figure out why torch.utils.data.ChainDataset is broken
            self.test_dataset = ChainedDataset(
                SceneFlowDataset(self.root_dir, "test_15mm", self.grayscale),
                SceneFlowDataset(self.root_dir, "test_35mm", self.grayscale),
            )

    def dataloader(self, dataset: SceneFlowDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.dataloader(self.validation_dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.dataloader(self.test_dataset)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.grayscale:
            mean = sum(mean) / len(mean)
            std = sum(std) / len(std)

        batch["image_left"] = resize(normalize(batch["image_left"], mean, std), self.res, antialias=False)
        batch["image_right"] = resize(normalize(batch["image_right"], mean, std), self.res, antialias=False)
        batch["disparity"] = resize(batch["disparity"], self.res, antialias=False)

        return batch


if __name__ == "__main__":
    from lovely_tensors import lovely

    datamodule = SceneFlowDataModule(res=(224, 224), root_dir=Path("/media/jo/Flash/datasets/sceneflow"), batch_size=24)
    datamodule.prepare_data()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))

    print(f"Sample batch loaded with batch_size {datamodule.batch_size}")
    print(f"Left Image: {lovely(batch['image_left'])}")
    print(f"Right Image: {lovely(batch['image_right'])}")
    print(f"Disparity Map: {lovely(batch['disparity'])}")
    print(f"Focal Length: {lovely(batch['focal_length'])}")
    print(f"Total Size: {sum(p.numel() for p in batch.values())}")
