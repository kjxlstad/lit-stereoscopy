from pathlib import Path
from typing import Literal, get_args
import logging

DataPath = tuple[Path, Path, Path]
DataSplit = Literal["train", "val", "test_15mm", "test_35mm"]

SAVE_PATHS = {split: Path(__file__).parent / f"{split}_paths.dat" for split in get_args(DataSplit)}


def _valid_data_paths(root_dir: Path, split: DataSplit) -> list[DataPath]:
    if split not in get_args(DataSplit):
        raise ValueError(f"split must be one of {get_args(DataSplit)}")

    if split in ("train", "val"):
        img_dir = root_dir / split / "images"
        disp_dir = root_dir / split / "disparity"

        # Match each path up with a unique key, join subfolder and file-name
        left_images = {f"{p.parent.parent.name}-{p.stem}": p for p in img_dir.glob("[ABC]/*/left/*.webp")}
        right_images = {f"{p.parent.parent.name}-{p.stem}": p for p in img_dir.glob("[ABC]/*/right/*.webp")}
        disparity = {f"{p.parent.name}-{p.stem}": p for p in disp_dir.glob("[ABC]/*/*.pfm")}
    else:
        sub_dir = root_dir / f"test_{split.split('_')[1][:-2]}mm"

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


def data_paths(root_dir: Path, split: DataSplit) -> list[DataPath]:
    """Fetches paths to data in sceneflow dataset.

    Uses a cached list of paths if it exists, otherwise generates the list and saves it.

    Args:
        root_dir: Path to root directory of sceneflow dataset.
        split: "train | val | test_15mm | test_35mm", which split to fetch paths for.

    Returns:
        A list containing tuples of paths to left image, right image and disparity map.
    """
    if not root_dir.is_dir():
        raise ValueError(f"root_dir must be a directory containing sceneflow dataset, got {root_dir}")

    cache_file = SAVE_PATHS[split]

    if not SAVE_PATHS[split].exists():
        logging.info(f"Path cache for dataset {split} split not found, generating...")
        data_paths = _valid_data_paths(root_dir, split)

        logging.info(f"Saving paths for dataset {split} split to {SAVE_PATHS[split]}")
        _write_paths(data_paths, cache_file, root_dir)

        return data_paths

    return _read_paths(cache_file, root_dir)


def cache_all(root_dir: Path) -> None:
    """Caches all dataset paths to disk."""
    for split in get_args(DataSplit):
        data_paths(root_dir, split)
