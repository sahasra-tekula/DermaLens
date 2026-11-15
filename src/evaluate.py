"""Validation-only model selection, followed by one frozen test evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data_pipeline import (
    CLASS_TO_IDX,
    CLASSES,
    DermImageDataset,
    PROJECT_ROOT,
    SPLIT_DIR,
    _seed_worker,
    load_config,
)
from src.train import (
    BATCH_SIZE,
    NUM_WORKERS,
    build_pretrained_resnet,
    configure_torch_home,
    seed_everything,
)


METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
MEL_IDX = CLASS_TO_IDX["mel"]


def _load_stage_model(stage: str, checkpoint_path: Path, config: dict[str, Any]) -> torch.nn.Module:
    configure_torch_home(config["checkpoint_root"])
    model = build_pretrained_resnet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if stage == "stage_a":
        frozen_state = checkpoint.get("frozen_backbone_state_dict")
        if frozen_state is not None:
            model.load_state_dict(frozen_state, strict=False)
        model.fc.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict(
    model: torch.nn.Module,
    split_name: str,
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dataset = DermImageDataset(SPLIT_DIR / f"{split_name}.csv", cache_root, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(42),
    )
    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(loader, start=1):
            probabilities.append(torch.softmax(model(images), dim=1).cpu().numpy())
            labels.extend(batch_labels.tolist())
            if batch_index % 100 == 0:
                print(f"{split_name} inference batch {batch_index}/{len(loader)}", flush=True)
    return np.concatenate(probabilities), np.asarray(labels, dtype=np.int64), dataset.frame["image_id"].astype(str).tolist()


def _thresholded_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    non_melanoma = probabilities.copy()
    non_melanoma[:, MEL_IDX] = -1.0
    predictions = non_melanoma.argmax(axis=1)
    predictions[probabilities[:, MEL_IDX] >= threshold] = MEL_IDX
    return predictions


def select_model_and_threshold(config: dict[str, Any]) -> dict[str, Any]:
    """Compare stages and select a melanoma threshold using validation data only."""
    stage_candidates = []
    for stage in ("stage_a", "stage_b"):
        checkpoint_path = config["checkpoint_root"] / stage / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing {stage} validation checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metric_row = checkpoint["metrics"]
        stage_candidates.append(
            {
                "stage": stage,
                "checkpoint_path": str(checkpoint_path),
                "best_epoch": int(checkpoint["best_epoch"]),
                "val_macro_f1": float(checkpoint["best_val_macro_f1"]),
                "val_mel_recall": float(metric_row["val_mel_recall"]),
            }
        )
    # Macro-F1 is primary; melanoma recall breaks an exact macro-F1 tie.
    selected = max(stage_candidates, key=lambda item: (item["val_macro_f1"], item["val_mel_recall"]))
    model = _load_stage_model(selected["stage"], Path(selected["checkpoint_path"]), config)
    val_probabilities, val_labels, _ = _predict(model, "val", config["cache_root"])
    val_melanoma = (val_labels == MEL_IDX).astype(np.uint8)
    precision, recall, thresholds = precision_recall_curve(val_melanoma, val_probabilities[:, MEL_IDX])
    if len(thresholds) == 0:
        threshold = 0.5
        threshold_index = 0
    else:
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        threshold_index = int(np.argmax(f1_values))
        threshold = float(thresholds[threshold_index])
    val_threshold_predictions = _thresholded_predictions(val_probabilities, threshold)
    val_argmax_predictions = val_probabilities.argmax(axis=1)
    threshold_metrics = {
        "threshold": threshold,
        "precision": float(precision_score(val_melanoma, val_threshold_predictions == MEL_IDX, zero_division=0)),
        "recall": float(recall_score(val_melanoma, val_threshold_predictions == MEL_IDX, zero_division=0)),
        "f1": float(f1_score(val_melanoma, val_threshold_predictions == MEL_IDX, zero_division=0)),
        "argmax_precision": float(precision_score(val_melanoma, val_argmax_predictions == MEL_IDX, zero_division=0)),
        "argmax_recall": float(recall_score(val_melanoma, val_argmax_predictions == MEL_IDX, zero_division=0)),
    }
    selection = {
        "selected_stage": selected["stage"],
        "selected_checkpoint": selected["checkpoint_path"],
        "selection_rule": "highest validation macro-F1; validation melanoma recall breaks exact ties",
        "stage_comparison": stage_candidates,
        "selected_validation_threshold_metrics": threshold_metrics,
        "threshold_rule": "maximize binary melanoma F1 on validation precision-recall curve",
        "test_evaluated": False,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(stage_candidates).to_csv(METRICS_DIR / "stage_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(recall, precision, color="#527a8a")
    axes[0].set(title="Validation melanoma precision-recall", xlabel="Recall", ylabel="Precision")
    axes[0].grid(alpha=0.25)
    axes[1].plot(thresholds, precision[:-1], label="precision")
    axes[1].plot(thresholds, recall[:-1], label="recall")
    axes[1].axvline(threshold, color="black", linestyle="--", label=f"selected={threshold:.3f}")
    axes[1].set(title="Threshold selection on validation", xlabel="Melanoma probability threshold", ylabel="Score")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(METRICS_DIR / "melanoma_threshold_validation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def _plot_confusion(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES, yticklabels=CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test confusion matrix (7-class argmax)")
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_roc(y_hot: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_hot[:, index], probabilities[:, index])
        auc = roc_auc_score(y_hot[:, index], probabilities[:, index])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test one-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pr(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float, path: Path
) -> None:
    binary = (y_true == MEL_IDX).astype(np.uint8)
    precision, recall, thresholds = precision_recall_curve(binary, probabilities[:, MEL_IDX])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#9a5b49")
    if len(thresholds):
        index = int(np.argmin(np.abs(thresholds - threshold)))
        ax.scatter(recall[index], precision[index], marker="o", color="black", label=f"validation threshold={threshold:.3f}")
        ax.legend()
    ax.set(xlabel="Melanoma recall", ylabel="Melanoma precision", title="Test melanoma precision-recall curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_missed_case_montages(
    missed: pd.DataFrame, cache_root: Path, output_root: Path, per_page: int = 24
) -> list[str]:
    if missed.empty:
        return []
    output_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for page, start in enumerate(range(0, len(missed), per_page), start=1):
        subset = missed.iloc[start : start + per_page]
        cols = 6
        rows = int(np.ceil(len(subset) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, row) in zip(axes.flat, subset.iterrows()):
            with Image.open(cache_root / f"{row['image_id']}.jpg") as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"{row['image_id']}\np(mel)={row['p_mel']:.3f}", fontsize=7)
            ax.axis("off")
        fig.suptitle("Test melanomas missed at the validation-selected threshold")
        fig.tight_layout()
        destination = output_root / f"missed_melanomas_page_{page}.png"
        fig.savefig(destination, dpi=130)
        plt.close(fig)
        paths.append(str(destination))
    return paths


def evaluate_test_once(config: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the held-out test set once, after stage and threshold are frozen."""
    metrics_path = METRICS_DIR / "metrics.json"
    started_path = METRICS_DIR / "test_evaluation_started.json"
    if metrics_path.exists() or started_path.exists():
        raise FileExistsError("A test evaluation has already started; refusing to read the test set again")
    if not selection.get("selected_checkpoint") or "selected_validation_threshold_metrics" not in selection:
        raise ValueError("Freeze the model and validation threshold with --mode select first")
    threshold = float(selection["selected_validation_threshold_metrics"]["threshold"])
    stage = selection["selected_stage"]
    checkpoint_path = Path(selection["selected_checkpoint"])
    started_record = {
        "started_unix": time.time(),
        "selected_stage": stage,
        "selected_checkpoint": str(checkpoint_path),
        "validation_threshold": threshold,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    started_path.write_text(json.dumps(started_record, indent=2) + "\n", encoding="utf-8")

    model = _load_stage_model(stage, checkpoint_path, config)
    # This is the first point in Phase 3 that opens test images.
    probabilities, y_true, image_ids = _predict(model, "test", config["cache_root"])
    y_pred = probabilities.argmax(axis=1)
    y_thresholded = _thresholded_predictions(probabilities, threshold)
    labels = np.arange(len(CLASSES))
    y_hot = np.eye(len(CLASSES), dtype=np.uint8)[y_true]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    auc_by_class = roc_auc_score(y_hot, probabilities, average=None)
    auc_macro = roc_auc_score(y_hot, probabilities, average="macro")
    counts = pd.Series(y_true).map({index: name for name, index in CLASS_TO_IDX.items()}).value_counts()
    counts_by_class = {name: int(counts.get(name, 0)) for name in CLASSES}

    rng = np.random.default_rng(42)
    bootstrap_scores = []
    for _ in range(2000):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        bootstrap_scores.append(
            f1_score(
                y_true[indices],
                y_pred[indices],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    ci_low, ci_high = np.percentile(bootstrap_scores, [2.5, 97.5])

    mel_binary = (y_true == MEL_IDX).astype(np.uint8)
    mel_argmax = y_pred == MEL_IDX
    mel_thresholded = y_thresholded == MEL_IDX
    baseline_pred = np.full_like(y_true, CLASS_TO_IDX["nv"])
    baseline_report = classification_report(
        y_true,
        baseline_pred,
        labels=labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "selected_stage": stage,
        "selected_checkpoint": str(checkpoint_path),
        "melanoma_probability_threshold": threshold,
        "test_evaluation_count": 1,
        "test_image_count": int(len(y_true)),
        "test_class_counts": counts_by_class,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "macro_f1_bootstrap_resamples": 2000,
        "macro_f1_bootstrap_seed": 42,
        "macro_f1_bootstrap_method": "unstratified image resampling with replacement; fixed seven-class label set",
        "per_class_ovr_roc_auc": {name: float(auc_by_class[index]) for index, name in enumerate(CLASSES)},
        "macro_ovr_roc_auc": float(auc_macro),
        "melanoma_argmax_precision": float(precision_score(mel_binary, mel_argmax, zero_division=0)),
        "melanoma_argmax_recall": float(recall_score(mel_binary, mel_argmax, zero_division=0)),
        "melanoma_argmax_f1": float(f1_score(mel_binary, mel_argmax, zero_division=0)),
        "melanoma_threshold_precision": float(precision_score(mel_binary, mel_thresholded, zero_division=0)),
        "melanoma_threshold_recall": float(recall_score(mel_binary, mel_thresholded, zero_division=0)),
        "melanoma_threshold_f1": float(f1_score(mel_binary, mel_thresholded, zero_division=0)),
        "per_class_metrics_argmax": class_report,
        "always_predict_nv_baseline": {
            "accuracy": float(accuracy_score(y_true, baseline_pred)),
            "macro_precision": float(precision_score(y_true, baseline_pred, labels=labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, baseline_pred, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, baseline_pred, labels=labels, average="macro", zero_division=0)),
            "per_class_metrics": baseline_report,
        },
        "split_note": "test was evaluated once after model and melanoma threshold selection on validation",
    }

    pd.DataFrame(class_report).T.to_csv(METRICS_DIR / "classification_report.csv")
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(METRICS_DIR / "confusion_matrix.csv")
    _plot_confusion(matrix, METRICS_DIR / "confusion_matrix.png")
    _plot_roc(y_hot, probabilities, METRICS_DIR / "roc_ovr.png")
    _plot_pr(y_true, probabilities, threshold, METRICS_DIR / "melanoma_pr_test.png")

    missed = pd.DataFrame(
        {
            "image_id": image_ids,
            "true_dx": [CLASSES[index] for index in y_true],
            "argmax_dx": [CLASSES[index] for index in y_pred],
            "threshold_dx": [CLASSES[index] for index in y_thresholded],
            "p_mel": probabilities[:, MEL_IDX],
        }
    )
    missed = missed[(missed["true_dx"] == "mel") & (missed["threshold_dx"] != "mel")].copy()
    missed.to_csv(METRICS_DIR / "missed_melanoma_cases.csv", index=False)
    metrics["missed_melanoma_count_at_selected_threshold"] = int(len(missed))
    metrics["missed_melanoma_case_ids"] = missed["image_id"].tolist()
    metrics["test_montage_paths_external"] = _save_missed_case_montages(
        missed, config["cache_root"], config["cache_root"] / "test_error_review"
    )
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    selection["test_evaluated"] = True
    (METRICS_DIR / "model_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"per_class_metrics_argmax", "always_predict_nv_baseline"}}, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to dataset_config.json")
    parser.add_argument("--mode", choices=("select", "test"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    if args.mode == "select":
        select_model_and_threshold(config)
    else:
        selection_path = METRICS_DIR / "model_selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError("Run --mode select first; test set has not been unlocked")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        evaluate_test_once(config, selection)


if __name__ == "__main__":
    main()
