from pathlib import Path
from typing import Literal
import logging

DataPath = tuple[Path, Path, Path]

SAVE_PATHS = {
    "train": Path(__file__).parent / "train_paths.dat",
    "val": Path(__file__).parent / "val_paths.dat",
    "test_15mm": Path(__file__).parent / "test_15mm_paths.dat",
    "test_35mm": Path(__file__).parent / "test_35mm_paths.dat",
}


def _valid_train_paths(root_dir: Path, split: Literal["train", "val"]) -> list[DataPath]:
    if split not in ("train", "val"):
        raise ValueError("stage must be one of 'train' or 'val'")

    img_dir = root_dir / split / "images"
    disp_dir = root_dir / split / "disparity"

    # Match each path up with a unique key, join subfolder and file-name
    left_images = {f"{p.parent.parent.name}-{p.stem}": p for p in img_dir.glob("[ABC]/*/left/*.webp")}
    right_images = {f"{p.parent.parent.name}-{p.stem}": p for p in img_dir.glob("[ABC]/*/right/*.webp")}
    disparity = {f"{p.parent.name}-{p.stem}": p for p in disp_dir.glob("[ABC]/*/*.pfm")}

    # All keys that are in all three dictionaries, e.g., every instance where all three files exist
    return [
        (left_images[key], right_images[key], disparity[key])
        for key in left_images.keys() & right_images.keys() & disparity.keys()
    ]


def _valid_test_paths(root_dir: Path, focal_length: Literal[15, 35]) -> list[DataPath]:
    if focal_length not in (15, 35):
        raise ValueError("focal_legnth must be one of 15 or 35")

    sub_dir = root_dir / f"test_{focal_length}mm"

    # Match each path up with a unique key, join subfolder and file-name
    left_images = {p.stem: p for p in (sub_dir / "images" / "left").glob("*.webp")}
    right_images = {p.stem: p for p in (sub_dir / "images" / "right").glob("*.webp")}
    disparity = {p.stem: p for p in (sub_dir / "disparity").glob("*.pfm")}

    # All keys that are in all three dictionaries, e.g., every instance where all three files exist
    return [
        (left_images[key], right_images[key], disparity[key])
        for key in left_images.keys() & right_images.keys() & disparity.keys()
    ]


def _write_paths(paths: list[DataPath], savefile: Path, root: Path) -> None:
    with open(savefile, "w") as f:
        for left, right, disp in paths:
            rel_paths = map(lambda p: str(p.relative_to(root)), (left, right, disp))
            f.write(" ".join(rel_paths) + "\n")


def _read_paths(savefile: Path, root: Path) -> list[DataPath]:
    with open(savefile, "r") as f:
        rel_paths = [line.split() for line in f.readlines()]
        return [(root / l, root / r, root / d) for l, r, d in rel_paths]


def train_paths(root_dir: Path, split: Literal["train", "val"]) -> list[DataPath]:
    """Fetches paths to training data.

    Uses a cached list of paths if it exists, otherwise generates the list and saves it.

    Args:
        root_dir: Path to root directory of sceneflow dataset.
        split: "train | val", which split to fetch paths for.

    Returns:
        A list containing tuples of paths to left image, right image and disparity map.
    """
    if not root_dir.is_dir:
        raise ValueError(f"root_dir must be a directory containing sceneflow dataset, got {root_dir}")

    cache_file = SAVE_PATHS[split]

    if not SAVE_PATHS[split].exists():
        logging.info(f"Path cache for {split} split dataset not found, generating...")
        data_paths = _valid_train_paths(root_dir, split)

        logging.info(f"Saving paths for {split} split dataset to {SAVE_PATHS[split]}")
        _write_paths(data_paths, cache_file, root_dir)

        return data_paths

    return _read_paths(cache_file, root_dir)


def test_paths(root_dir: Path, focal_length: Literal[15, 35]):
    """Fetches paths to training data.

    Uses a cached list of paths if it exists, otherwise generates the list and saves it.


    Args:
        root_dir: Path to root directory of sceneflow dataset.
        split: "train | val", which split to fetch paths for.

    Returns:
        A list containing tuples of paths to left image, right image and disparity map.
    """
    if not root_dir.is_dir:
        raise ValueError(f"root_dir must be a directory containing sceneflow dataset, got {root_dir}")

    cache_file = SAVE_PATHS[f"test_{focal_length}mm"]

    if not SAVE_PATHS[f"test_{focal_length}mm"].exists():
        logging.info(f"Path cache for {focal_length}mm focal length test set not found, generating...")
        data_paths = _valid_test_paths(root_dir, focal_length)

        logging.info(f"Saving paths for {focal_length}mm test set focal_length to {cache_file}")
        _write_paths(data_paths, cache_file, root_dir)

        return data_paths

    return _read_paths(cache_file, root_dir)


def cache_all(root_dir: Path) -> None:
    train_paths(root_dir, "train")
    train_paths(root_dir, "val")
    test_paths(root_dir, 15)
    test_paths(root_dir, 35)
