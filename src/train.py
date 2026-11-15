"""CPU training for the two-stage ResNet18 HAM10000 experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.data_pipeline import (
    CLASS_TO_IDX,
    CLASSES,
    DermImageDataset,
    PROJECT_ROOT,
    SPLIT_DIR,
    _seed_worker,
    load_config,
)


METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
DEFAULT_STAGE_A_EPOCHS = 30
DEFAULT_STAGE_B_EPOCHS = 6
STAGE_A_PATIENCE = 5
STAGE_B_PATIENCE = 3
BATCH_SIZE = 16
NUM_WORKERS = 2
CPU_THREADS = 10
MEL_IDX = CLASS_TO_IDX["mel"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch_home(checkpoint_root: Path) -> None:
    """Keep downloaded pretrained weights outside the repo and OneDrive."""
    torch_home = checkpoint_root / "torch_home"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)


def build_pretrained_resnet() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def compute_train_class_weights(train_csv: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute balanced cross-entropy weights from train.csv only."""
    labels = pd.read_csv(train_csv)["dx"]
    counts = labels.value_counts().reindex(CLASSES, fill_value=0)
    if (counts == 0).any():
        raise ValueError("Every class must occur in the training split")
    weights = len(labels) / (len(CLASSES) * counts.to_numpy(dtype=np.float64))
    result = {
        "source": "splits/train.csv only",
        "class_counts": {name: int(counts[name]) for name in CLASSES},
        "weights": {name: float(weight) for name, weight in zip(CLASSES, weights)},
    }
    return torch.tensor(weights, dtype=torch.float32), result


def _feature_forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the 512-dimensional ResNet18 representation before its final layer."""
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def _split_signature(split_csv: Path) -> str:
    import hashlib

    return hashlib.sha256(split_csv.read_bytes()).hexdigest()


def extract_feature_cache(
    model: nn.Module,
    split_name: str,
    cache_root: Path,
    checkpoint_root: Path,
    seed: int,
) -> tuple[Path, Path]:
    """Extract and cache frozen ResNet18 features for train or validation only."""
    if split_name not in {"train", "val"}:
        raise ValueError("Stage A feature extraction is restricted to train and validation")
    split_csv = SPLIT_DIR / f"{split_name}.csv"
    split_frame = pd.read_csv(split_csv)
    feature_dir = cache_root / "resnet18_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{split_name}.npy"
    label_path = feature_dir / f"{split_name}_labels.npy"
    meta_path = feature_dir / f"{split_name}.json"
    signature = _split_signature(split_csv)
    expected_meta = {
        "split_sha256": signature,
        "weights": ResNet18_Weights.DEFAULT.name,
        "feature_dim": 512,
        "rows": len(split_frame),
    }
    if feature_path.is_file() and label_path.is_file() and meta_path.is_file():
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta == expected_meta:
            return feature_path, label_path

    dataset = DermImageDataset(split_csv, cache_root, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    temp_feature = feature_path.with_suffix(".tmp.npy")
    features = np.lib.format.open_memmap(
        temp_feature, mode="w+", dtype=np.float32, shape=(len(dataset), 512)
    )
    model.eval()
    offset = 0
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader, start=1):
            batch_features = _feature_forward(model, images).cpu().numpy()
            features[offset : offset + len(batch_features)] = batch_features
            offset += len(batch_features)
            if batch_index % 100 == 0:
                print(f"feature cache {split_name}: {offset}/{len(dataset)}", flush=True)
    features.flush()
    del features
    temp_feature.replace(feature_path)
    labels = dataset.frame["dx"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
    np.save(label_path, labels)
    meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")
    print(f"feature cache {split_name}: {offset}/{len(dataset)} complete", flush=True)
    return feature_path, label_path


class CachedFeatureDataset(Dataset):
    def __init__(self, feature_path: Path, label_path: Path):
        self.features = np.load(feature_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = torch.from_numpy(np.array(self.features[index], copy=True))
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return feature, label


def _macro_f1_and_mel_recall(y_true: list[int], y_pred: list[int]) -> tuple[float, float]:
    macro_f1 = f1_score(y_true, y_pred, labels=range(len(CLASSES)), average="macro", zero_division=0)
    mel_recall = recall_score(y_true, y_pred, labels=[MEL_IDX], average=None, zero_division=0)[0]
    return float(macro_f1), float(mel_recall)


def _evaluate_head(
    head: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float, float]:
    head.eval()
    total_loss = 0.0
    total = 0
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for features, labels in loader:
            logits = head(features)
            loss = criterion(logits, labels)
            total_loss += float(loss) * len(labels)
            total += len(labels)
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).tolist())
    macro_f1, mel_recall = _macro_f1_and_mel_recall(actual, predicted)
    return total_loss / total, macro_f1, mel_recall


def _evaluate_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss) * len(labels)
            total += len(labels)
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).tolist())
            if batch_index % 100 == 0:
                print(f"validation batch {batch_index}/{len(loader)}", flush=True)
    macro_f1, mel_recall = _macro_f1_and_mel_recall(actual, predicted)
    return total_loss / total, macro_f1, mel_recall


def _train_head_epoch(
    head: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    head.train()
    total_loss = 0.0
    total = 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total += len(labels)
    return total_loss / total


def _train_model_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    model.train()
    # Keep every frozen backbone block, including its BatchNorm layers, in eval mode.
    for module in (model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3):
        module.eval()
    model.layer4.train()
    model.fc.train()
    total_loss = 0.0
    total = 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total += len(labels)
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"train batch {batch_index}/{len(loader)}", flush=True)
    return total_loss / total


def _rng_state(train_loader: DataLoader) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": train_loader.generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_loader.generator.set_state(state["loader"])


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
    if not history:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], marker="o")
    axes[1].set_title("Validation macro-F1")
    axes[2].plot(epochs, [row["val_mel_recall"] for row in history], marker="o")
    axes[2].set_title("Validation melanoma recall")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def _checkpoint_state(
    stage: str,
    model_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_f1: float,
    best_epoch: int,
    stale_epochs: int,
    train_loader: DataLoader,
    class_weights: torch.Tensor,
    seed: int,
    metric_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "seeds": {"python": seed, "numpy": seed, "torch": seed, "dataloader": seed},
        "rng_state": _rng_state(train_loader),
        "class_weights_train_only": class_weights.cpu(),
        "metrics": metric_row,
        "weights": ResNet18_Weights.DEFAULT.name,
    }


def run_stage_a(config: dict[str, Any], args: argparse.Namespace, class_weights: torch.Tensor) -> None:
    seed = int(config["seed"])
    stage_dir = config["checkpoint_root"] / "stage_a"
    stage_dir.mkdir(parents=True, exist_ok=True)
    history_path = METRICS_DIR / "stage_a_history.csv"
    configure_torch_home(config["checkpoint_root"])
    model = build_pretrained_resnet()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()  # Backbone and BatchNorm remain frozen in eval mode for feature extraction.
    model.fc = nn.Identity()

    feature_paths = {
        split: extract_feature_cache(
            model, split, config["cache_root"], config["checkpoint_root"], seed
        )
        for split in ("train", "val")
    }
    train_data = CachedFeatureDataset(*feature_paths["train"])
    val_data = CachedFeatureDataset(*feature_paths["val"])
    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=0)
    head = nn.Linear(512, len(CLASSES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = _read_history(history_path)
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    last_path = stage_dir / "last.pt"

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume Stage A; missing {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        head.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_f1 = float(checkpoint["best_val_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng_state"], train_loader)

    max_epochs = args.epochs if args.epochs is not None else DEFAULT_STAGE_A_EPOCHS
    new_epoch_limit = args.max_new_epochs if args.max_new_epochs is not None else max_epochs
    stop_at = min(max_epochs, start_epoch - 1 + new_epoch_limit)
    for epoch in range(start_epoch, stop_at + 1):
        started = time.perf_counter()
        train_loss = _train_head_epoch(head, train_loader, criterion, optimizer)
        val_loss, val_f1, val_mel_recall = _evaluate_head(head, val_loader, criterion)
        scheduler.step(val_f1)
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_mel_recall": val_mel_recall,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        _write_history(history_path, history)
        state = _checkpoint_state(
            "stage_a", head.state_dict(), optimizer, scheduler, epoch, best_val_f1,
            best_epoch, stale_epochs, train_loader, class_weights, seed, row,
        )
        # Stage A trains only the head, but the checkpoint also carries its frozen backbone.
        state["frozen_backbone_state_dict"] = model.state_dict()
        _save_checkpoint(last_path, state)
        if improved:
            _save_checkpoint(stage_dir / "best.pt", state)
        print(
            f"stage_a epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_macro_f1={val_f1:.5f} val_mel_recall={val_mel_recall:.5f} "
            f"seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        if stale_epochs >= STAGE_A_PATIENCE:
            print("stage_a early stopping on validation macro-F1", flush=True)
            break
    print(f"stage_a best_epoch={best_epoch} best_val_macro_f1={best_val_f1:.6f}", flush=True)


def _freeze_except_layer4(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.layer4.parameters():
        parameter.requires_grad_(True)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)


def run_stage_b(config: dict[str, Any], args: argparse.Namespace, class_weights: torch.Tensor) -> None:
    seed = int(config["seed"])
    stage_dir = config["checkpoint_root"] / "stage_b"
    stage_dir.mkdir(parents=True, exist_ok=True)
    history_path = METRICS_DIR / "stage_b_history.csv"
    configure_torch_home(config["checkpoint_root"])
    model = build_pretrained_resnet()
    _freeze_except_layer4(model)
    train_dataset = DermImageDataset(SPLIT_DIR / "train.csv", config["cache_root"], augment=True)
    val_dataset = DermImageDataset(SPLIT_DIR / "val.csv", config["cache_root"], augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history = _read_history(history_path)
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    last_path = stage_dir / "last.pt"

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume Stage B; missing {last_path}")
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_f1 = float(checkpoint["best_val_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng_state"], train_loader)
    else:
        stage_a_best = config["checkpoint_root"] / "stage_a" / "best.pt"
        if not stage_a_best.is_file():
            raise FileNotFoundError(f"Train Stage A first; missing {stage_a_best}")
        stage_a_checkpoint = torch.load(stage_a_best, map_location="cpu", weights_only=False)
        model.fc.load_state_dict(stage_a_checkpoint["model_state_dict"])

    max_epochs = args.epochs if args.epochs is not None else DEFAULT_STAGE_B_EPOCHS
    new_epoch_limit = args.max_new_epochs if args.max_new_epochs is not None else max_epochs
    stop_at = min(max_epochs, start_epoch - 1 + new_epoch_limit)
    for epoch in range(start_epoch, stop_at + 1):
        started = time.perf_counter()
        train_loss = _train_model_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, val_mel_recall = _evaluate_model(model, val_loader, criterion)
        scheduler.step(val_f1)
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_mel_recall": val_mel_recall,
            "elapsed_seconds": time.perf_counter() - started,
            "layer4_learning_rate": optimizer.param_groups[0]["lr"],
            "head_learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        _write_history(history_path, history)
        state = _checkpoint_state(
            "stage_b", model.state_dict(), optimizer, scheduler, epoch, best_val_f1,
            best_epoch, stale_epochs, train_loader, class_weights, seed, row,
        )
        _save_checkpoint(last_path, state)
        if improved:
            _save_checkpoint(stage_dir / "best.pt", state)
        print(
            f"stage_b epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_macro_f1={val_f1:.5f} val_mel_recall={val_mel_recall:.5f} "
            f"seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        if stale_epochs >= STAGE_B_PATIENCE:
            print("stage_b early stopping on validation macro-F1", flush=True)
            break
    print(f"stage_b best_epoch={best_epoch} best_val_macro_f1={best_val_f1:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to dataset_config.json")
    parser.add_argument("--stage", choices=("a", "b", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Resume the selected stage from its last.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Maximum total epochs for selected stages")
    parser.add_argument("--max-new-epochs", type=int, default=None, help="Limit epochs this invocation (useful for timing/resume)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    weights, weights_report = compute_train_class_weights(SPLIT_DIR / "train.csv")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "class_weights.json").write_text(
        json.dumps(weights_report, indent=2) + "\n", encoding="utf-8"
    )
    print("device=cpu; torch_threads=10; batch_size=16; num_workers=2", flush=True)
    print("class weights (computed from train split only): " + json.dumps(weights_report["weights"]), flush=True)
    if args.stage in {"a", "all"}:
        run_stage_a(config, args, weights)
    if args.stage in {"b", "all"}:
        run_stage_b(config, args, weights)


if __name__ == "__main__":
    main()
