import re
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from data import data_paths, DataSplit, DataPath


def load_image(path: Path) -> torch.Tensor:
    array = np.array(Image.open(path)).astype(np.float32) / 255
    return torch.tensor(array).permute(2, 0, 1)


def load_disparity_map(path: Path) -> torch.Tensor:
    with open(path, "rb") as pfm_file:
        next(pfm_file)  # skip header

        dim_line = pfm_file.readline().decode("utf-8")
        width, height = map(int, re.match(r"^(\d+)\s(\d+)\s$", dim_line).groups())  # type: ignore

        scale = float(pfm_file.readline().decode("utf-8").rstrip())
        endianness = "<f" if scale < 0 else ">f"

        data = np.fromfile(pfm_file, endianness).reshape(height, width)

    data = np.flipud(data)
    data = np.ascontiguousarray(data)

    return torch.tensor(data).unsqueeze(0)


class SceneFlowDataset(Dataset):
    def __init__(self, root_dir: Path, split: DataSplit, grayscale: bool = False):
        self.root_dir = root_dir
        self.grayscale = grayscale
        self.focal_length = int(split[5:7]) if "test" in split else None

        self.paths: list[DataPath] = data_paths(root_dir, split)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        image_left_path, image_right_path, disparity_path = self.paths[index]

        image_left = load_image(image_left_path)
        image_right = load_image(image_right_path)
        disparity = load_disparity_map(disparity_path)

        if self.grayscale:
            image_left = image_left.mean(dim=0, keepdim=True)
            image_right = image_right.mean(dim=0, keepdim=True)

        return image_left, image_right, disparity, self.focal_length


if __name__ == "__main__":
    from lovely_tensors import lovely

    dataset = SceneFlowDataset(Path("/media/jo/Flash/datasets/sceneflow"), "test_15mm")
    sample = next(iter(dataset))

    print(f"{dataset.__class__.__name__} dataset sample loaded")
    print(f"Left Image: {lovely(sample[0])}")
    print(f"Right Image: {lovely(sample[1])}")
    print(f"Disparity Map: {lovely(sample[2])}")
    print(f"Focal Length: {sample[3]}")
