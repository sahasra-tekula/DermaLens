"""HAM10000 manifest, lesion-level splits, resized image cache, and loaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}
REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_EXT = ".jpg"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load all dataset and cache locations from dataset_config.json."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "dataset_config.json"
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Copy dataset_config.example.json to "
            "dataset_config.json and set dataset_root and cache_root."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_root", "cache_root", "metadata_file", "image_directories", "image_size", "seed"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing keys: {sorted(missing)}")
    config["config_path"] = path
    config["dataset_root"] = Path(config["dataset_root"]).expanduser().resolve()
    config["cache_root"] = Path(config["cache_root"]).expanduser().resolve()
    config["metadata_path"] = config["dataset_root"] / config["metadata_file"]
    config["image_dirs"] = [config["dataset_root"] / item for item in config["image_directories"]]
    return config


def _ensure_cache_is_external(cache_root: Path, dataset_root: Path) -> None:
    cache_root = cache_root.resolve()
    dataset_root = dataset_root.resolve()
    if cache_root == dataset_root or dataset_root in cache_root.parents:
        raise ValueError("cache_root must not be inside the original dataset directory")
    if cache_root == PROJECT_ROOT or PROJECT_ROOT in cache_root.parents:
        raise ValueError("cache_root must be outside the project repository")
    if any(part.lower() == "onedrive" for part in cache_root.parts):
        raise ValueError("cache_root must be outside OneDrive")


def _image_path_map(image_dirs: list[Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for directory in image_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stem in paths:
                    raise ValueError(f"Image ID {path.stem} appears in more than one folder")
                paths[path.stem] = path
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Path], list[dict[str, str]], dict[str, Any]]:
    """Read metadata, validate image references, and remove exact byte duplicates."""
    if not config["metadata_path"].is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {config['metadata_path']}")
    metadata = pd.read_csv(config["metadata_path"])
    missing_columns = REQUIRED_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if metadata[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Metadata has missing image_id, lesion_id, or dx values")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate image_id rows")
    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["lesion_id"] = metadata["lesion_id"].astype(str)
    metadata["dx"] = metadata["dx"].astype(str)
    unknown = sorted(set(metadata["dx"]) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unexpected diagnosis labels: {unknown}")

    image_paths = _image_path_map(config["image_dirs"])
    metadata_ids = set(metadata["image_id"])
    missing_files = sorted(metadata_ids - set(image_paths))
    unreferenced_files = sorted(set(image_paths) - metadata_ids)
    if missing_files or unreferenced_files:
        raise ValueError(
            f"Image/metadata mismatch: missing files={len(missing_files)}, "
            f"unreferenced image files={len(unreferenced_files)}"
        )

    hashes: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(metadata_ids):
        hashes[_sha256(image_paths[image_id])].append(image_id)

    rows_by_id = metadata.set_index("image_id")
    removed_ids: set[str] = set()
    duplicate_records: list[dict[str, str]] = []
    for digest, image_ids in hashes.items():
        if len(image_ids) < 2:
            continue
        image_ids.sort()
        kept_id = image_ids[0]
        for duplicate_id in image_ids[1:]:
            kept_row = rows_by_id.loc[kept_id]
            duplicate_row = rows_by_id.loc[duplicate_id]
            if kept_row["dx"] != duplicate_row["dx"]:
                raise ValueError(f"Exact duplicate images have conflicting labels: {image_ids}")
            duplicate_records.append(
                {
                    "sha256": digest,
                    "kept_image_id": kept_id,
                    "excluded_image_id": duplicate_id,
                    "lesion_id": str(duplicate_row["lesion_id"]),
                    "dx": str(duplicate_row["dx"]),
                }
            )
            removed_ids.add(duplicate_id)

    manifest = metadata.loc[~metadata["image_id"].isin(removed_ids), ["image_id", "lesion_id", "dx"]]
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    audit = {
        "metadata_rows": int(len(metadata)),
        "image_files": int(len(image_paths)),
        "unique_lesion_ids": int(metadata["lesion_id"].nunique()),
        "missing_by_column": {str(k): int(v) for k, v in metadata.isna().sum().items()},
        "class_counts_before_deduplication": {
            name: int(metadata["dx"].value_counts().get(name, 0)) for name in CLASSES
        },
        "exact_duplicate_pairs_removed": int(len(duplicate_records)),
        "manifest_rows_after_deduplication": int(len(manifest)),
    }
    return manifest, {key: value for key, value in image_paths.items() if key not in removed_ids}, duplicate_records, audit


def make_grouped_splits(
    manifest: pd.DataFrame, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Make an approximate 70/15/15 split while keeping each lesion in one split."""
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(manifest, manifest["dx"], groups=manifest["lesion_id"]))
    train_val = manifest.iloc[train_val_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(train_val, train_val["dx"], groups=train_val["lesion_id"]))
    return {
        "train": train_val.iloc[train_idx].sort_values("image_id").reset_index(drop=True),
        "val": train_val.iloc[val_idx].sort_values("image_id").reset_index(drop=True),
        "test": manifest.iloc[test_idx].sort_values("image_id").reset_index(drop=True),
    }


def assert_no_lesion_overlap(splits: dict[str, pd.DataFrame]) -> None:
    group_sets = {name: set(frame["lesion_id"]) for name, frame in splits.items()}
    names = list(group_sets)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = group_sets[first] & group_sets[second]
            if overlap:
                raise AssertionError(f"Lesion IDs overlap between {first} and {second}: {sorted(overlap)[:5]}")


def _save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[["image_id", "lesion_id", "dx"]].to_csv(out_dir / f"{name}.csv", index=False)


def _report_split_counts(splits: dict[str, pd.DataFrame], report_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in splits.items():
        counts = frame["dx"].value_counts().reindex(CLASSES, fill_value=0)
        print(f"{split_name} ({len(frame)} images): " + ", ".join(f"{label}={counts[label]}" for label in CLASSES))
        rows.extend({"split": split_name, "dx": label, "count": int(counts[label])} for label in CLASSES)
    table = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "split_class_counts.csv", index=False)
    return table


def cache_resized_images(
    image_paths: dict[str, Path], cache_root: Path, dataset_root: Path, image_size: int = 224
) -> None:
    """Write individual resized JPEGs to an external cache, skipping valid cached files."""
    _ensure_cache_is_external(cache_root, dataset_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    total = len(image_paths)
    for index, (image_id, source_path) in enumerate(sorted(image_paths.items()), start=1):
        destination = cache_root / f"{image_id}{IMAGE_EXT}"
        if destination.is_file():
            try:
                with Image.open(destination) as cached:
                    if cached.size == (image_size, image_size):
                        continue
            except Exception:
                pass
        temporary = None
        try:
            with Image.open(source_path) as source:
                resized = source.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
            with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".jpg", delete=False) as handle:
                temporary = Path(handle.name)
            resized.save(temporary, format="JPEG", quality=95, subsampling=0)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        if index % 1000 == 0 or index == total:
            print(f"cached {index}/{total} resized images")


def _transforms(augment: bool) -> transforms.Compose:
    ops: list[Any] = []
    if augment:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class DermImageDataset(Dataset):
    """Load resized images from disk; only pixels are model input and dx is the target."""

    def __init__(self, split_csv: str | Path, cache_root: str | Path, augment: bool = False):
        self.frame = pd.read_csv(split_csv)
        missing = REQUIRED_COLUMNS - set(self.frame.columns)
        if missing:
            raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
        self.cache_root = Path(cache_root)
        self.transform = _transforms(augment)
        self.labels = self.frame["dx"].map(CLASS_TO_IDX)
        if self.labels.isna().any():
            raise ValueError("Split contains a diagnosis outside the fixed class mapping")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = self.cache_root / f"{row['image_id']}{IMAGE_EXT}"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (224, 224):
                raise ValueError(f"Cached image has unexpected size {image.size}: {image_path}")
            pixels = self.transform(image)
        return pixels, torch.tensor(int(self.labels.iloc[index]), dtype=torch.long)


def make_data_loaders(
    split_dir: str | Path = SPLIT_DIR,
    cache_root: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int | None = None,
) -> dict[str, DataLoader]:
    """Create image-only train/validation/test loaders reading the resized disk cache."""
    if cache_root is None:
        config = load_config()
        cache_root = config["cache_root"]
        if seed is None:
            seed = int(config["seed"])
    if seed is None:
        seed = 42
    loaders: dict[str, DataLoader] = {}
    for offset, name in enumerate(("train", "val", "test")):
        dataset = DermImageDataset(Path(split_dir) / f"{name}.csv", cache_root, augment=(name == "train"))
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=False,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed + offset),
        )
    return loaders


def _seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in each DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _save_eda(
    manifest: pd.DataFrame,
    cache_root: Path,
    audit: dict[str, Any],
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = manifest["dx"].value_counts().reindex(CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(CLASSES, [int(counts[name]) for name in CLASSES], color="#527a8a")
    ax.set(title="HAM10000 class counts after exact-duplicate removal", xlabel="Class", ylabel="Images")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(14, 3.2))
    for ax, label in zip(axes, CLASSES):
        sample_id = manifest.loc[manifest["dx"] == label, "image_id"].iloc[0]
        cached_path = cache_root / f"{sample_id}{IMAGE_EXT}"
        with Image.open(cached_path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("One resized HAM10000 example per class")
    fig.tight_layout()
    fig.savefig(cache_root / "sample_images_by_class.png", dpi=150)
    plt.close(fig)

    summary = {
        **audit,
        "manifest_class_counts_after_deduplication": {name: int(counts[name]) for name in CLASSES},
        "unique_lesions_after_deduplication": int(manifest["lesion_id"].nunique()),
        "manifest_rows": int(len(manifest)),
        "image_size": [224, 224],
        "model_inputs": "image pixels only; dx is the target; age, sex, localization, and dx_type are excluded",
    }
    (report_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(config_path: str | Path | None = None, build_cache: bool = True) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    manifest, image_paths, duplicate_records, audit = build_manifest(config)
    splits = make_grouped_splits(manifest, seed=int(config["seed"]))
    assert_no_lesion_overlap(splits)
    combined_ids = pd.concat(list(splits.values()))["image_id"]
    if combined_ids.duplicated().any() or set(combined_ids) != set(manifest["image_id"]):
        raise AssertionError("Splits must partition each de-duplicated manifest row exactly once")

    _save_splits(splits, SPLIT_DIR)
    _report_split_counts(splits, REPORT_DIR)
    duplicate_columns = ["sha256", "kept_image_id", "excluded_image_id", "lesion_id", "dx"]
    pd.DataFrame(duplicate_records, columns=duplicate_columns).to_csv(
        REPORT_DIR / "exact_duplicate_images.csv", index=False
    )
    if build_cache:
        cache_resized_images(
            image_paths,
            cache_root=config["cache_root"],
            dataset_root=config["dataset_root"],
            image_size=int(config["image_size"]),
        )
        _save_eda(manifest, config["cache_root"], audit, REPORT_DIR)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to dataset_config.json")
    parser.add_argument("--skip-cache", action="store_true", help="Write splits and reports without creating image cache")
    args = parser.parse_args()
    prepare_dataset(args.config, build_cache=not args.skip_cache)


if __name__ == "__main__":
    main()
