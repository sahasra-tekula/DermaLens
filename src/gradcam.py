"""Grad-CAM visualizations for the frozen DermaLens Stage B model.

The command-line workflow uses validation images only. It does not evaluate or
write to the held-out test split or the Phase 3 metric artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

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
MEL_IDX = CLASS_TO_IDX["mel"]


class GradCAM:
    """Grad-CAM using the output of the final block in ResNet18 layer4.

    A forward hook retains the target activation. Gradients are requested
    directly with ``torch.autograd.grad`` so model parameter gradients are not
    accumulated and no module backward hook is needed.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or model.layer4[-1]
        self._activation: torch.Tensor | None = None
        self._hook = self.target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return one Tensor")
        self._activation = output

    def __call__(
        self,
        images: torch.Tensor,
        target_classes: torch.Tensor | list[int] | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized heatmaps, logits, and target classes.

        Heatmaps are returned as ``(batch, input_height, input_width)`` floats
        in [0, 1]. If classes are omitted, each image's predicted class is
        explained.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected a BCHW image batch, received shape {tuple(images.shape)}")
        self.model.eval()
        self._activation = None

        # Explicitly disable inference mode as well as enabling autograd. This
        # keeps Grad-CAM usable when called from a surrounding inference block.
        with torch.inference_mode(False), torch.enable_grad():
            input_for_grad = images.detach().clone().requires_grad_(True)
            logits = self.model(input_for_grad)
            activation = self._activation
            if activation is None:
                raise RuntimeError("The Grad-CAM target layer did not run during the forward pass")
            if target_classes is None:
                classes = logits.argmax(dim=1)
            elif isinstance(target_classes, int):
                classes = torch.full(
                    (logits.shape[0],), target_classes, dtype=torch.long, device=logits.device
                )
            else:
                classes = torch.as_tensor(target_classes, dtype=torch.long, device=logits.device)
                if classes.ndim == 0:
                    classes = classes.expand(logits.shape[0])
            if classes.shape != (logits.shape[0],):
                raise ValueError("target_classes must contain one class index per image")
            if torch.any(classes < 0) or torch.any(classes >= logits.shape[1]):
                raise ValueError("target class index is outside the model output range")

            scores = logits.gather(1, classes[:, None]).sum()
            gradients = torch.autograd.grad(
                outputs=scores,
                inputs=activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            raw_cam = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
            raw_cam = F.interpolate(
                raw_cam,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            flat = raw_cam.flatten(start_dim=1)
            minimum = flat.min(dim=1).values[:, None, None, None]
            maximum = flat.max(dim=1).values[:, None, None, None]
            span = maximum - minimum
            normalized = torch.where(
                span > torch.finfo(raw_cam.dtype).eps,
                (raw_cam - minimum) / span.clamp_min(torch.finfo(raw_cam.dtype).eps),
                torch.zeros_like(raw_cam),
            )

        self._activation = None
        return normalized[:, 0].detach(), logits.detach(), classes.detach()

    def close(self) -> None:
        """Remove the forward hook and release the saved activation reference."""
        self._hook.remove()
        self._activation = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _load_selected_model(config: dict[str, Any]) -> tuple[nn.Module, float, Path]:
    selection_path = METRICS_DIR / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_stage") != "stage_b":
        raise ValueError("Grad-CAM is configured for the selected Stage B model")
    checkpoint_path = Path(selection["selected_checkpoint"])
    expected = config["checkpoint_root"] / "stage_b" / "best.pt"
    if checkpoint_path.resolve() != expected.resolve():
        raise ValueError(f"Selected checkpoint is not the Stage B best checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(selection["selected_validation_threshold_metrics"]["threshold"])
    return model, threshold, checkpoint_path


def _thresholded_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predictions = probabilities.copy()
    predictions[:, MEL_IDX] = -1.0
    result = predictions.argmax(axis=1)
    result[probabilities[:, MEL_IDX] >= threshold] = MEL_IDX
    return result


def _get_validation_predictions(
    model: nn.Module, cache_root: Path
) -> tuple[DermImageDataset, np.ndarray, np.ndarray]:
    dataset = DermImageDataset(SPLIT_DIR / "val.csv", cache_root, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(42),
    )
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, _labels in loader:
            batches.append(torch.softmax(model(images), dim=1).cpu().numpy())
    return dataset, np.concatenate(batches), dataset.labels.to_numpy(dtype=np.int64)


def _choose_examples(
    dataset: DermImageDataset,
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    per_group: int = 6,
) -> list[dict[str, Any]]:
    predictions = _thresholded_predictions(probabilities, threshold)
    rows = dataset.frame.reset_index(drop=True)
    records = [
        {
            "image_id": str(rows.iloc[i]["image_id"]),
            "true_idx": int(labels[i]),
            "pred_idx": int(predictions[i]),
            "p_mel": float(probabilities[i, MEL_IDX]),
            "correct": bool(labels[i] == predictions[i]),
        }
        for i in range(len(labels))
    ]
    correct_candidates = [item for item in records if item["correct"]]
    incorrect_candidates = [item for item in records if not item["correct"]]

    chosen_correct: list[dict[str, Any]] = []
    # Prefer a correct melanoma and then one example for each remaining class.
    for class_idx in range(len(CLASSES)):
        if len(chosen_correct) >= per_group:
            break
        candidate = next(
            (item for item in correct_candidates if item["true_idx"] == class_idx
             and item["image_id"] not in {chosen["image_id"] for chosen in chosen_correct}),
            None,
        )
        if candidate is not None:
            chosen_correct.append(candidate)
    for item in correct_candidates:
        if len(chosen_correct) >= per_group:
            break
        if item["image_id"] not in {chosen["image_id"] for chosen in chosen_correct}:
            chosen_correct.append(item)

    # Prefer three false-negative melanomas, then fill with other errors.
    missed_melanomas = [
        item for item in incorrect_candidates
        if item["true_idx"] == MEL_IDX and item["pred_idx"] != MEL_IDX
    ]
    if len(missed_melanomas) < min(3, per_group):
        raise RuntimeError(
            f"Validation has only {len(missed_melanomas)} threshold-missed melanomas; "
            "cannot include the requested variety of missed melanomas."
        )
    chosen_incorrect = missed_melanomas[: min(3, per_group)]
    for item in incorrect_candidates:
        if len(chosen_incorrect) >= per_group:
            break
        if item["image_id"] not in {chosen["image_id"] for chosen in chosen_incorrect}:
            chosen_incorrect.append(item)
    if len(chosen_correct) < per_group or len(chosen_incorrect) < per_group:
        raise RuntimeError("Validation split does not contain enough correct and incorrect predictions")
    for kind, chosen in (("correct", chosen_correct), ("incorrect", chosen_incorrect)):
        for item in chosen:
            item["kind"] = kind
    return chosen_correct + chosen_incorrect


def _write_panel(
    image_path: Path,
    heatmap: np.ndarray,
    record: dict[str, Any],
    output_path: Path,
    *,
    include_three_panels: bool,
) -> None:
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    colored = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = np.clip(0.55 * rgb + 0.45 * colored, 0, 1)
    true_name = CLASSES[record["true_idx"]]
    pred_name = CLASSES[record["pred_idx"]]
    status = "correct" if record["correct"] else "incorrect"
    if include_three_panels:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), dpi=120)
        axes[0].imshow(rgb)
        axes[0].set_title("Input")
        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"{record['image_id']} | true={true_name}, predicted={pred_name} ({status}) | "
            f"p(mel)={record['p_mel']:.3f}",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.round(overlay * 255))).save(
            output_path, format="JPEG", quality=82, optimize=True
        )


def generate_visualizations(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    include_readme_examples: bool = False,
) -> list[dict[str, Any]]:
    """Create six correct and six incorrect validation Grad-CAM panels."""
    config = load_config(config_path)
    model, threshold, checkpoint_path = _load_selected_model(config)
    dataset, probabilities, labels = _get_validation_predictions(model, config["cache_root"])
    examples = _choose_examples(dataset, probabilities, labels, threshold)
    destination = Path(output_dir) if output_dir else config["cache_root"].parent / "gradcam"
    destination = destination.resolve()
    if destination.drive.lower() != config["cache_root"].drive.lower():
        raise ValueError("Grad-CAM output must be stored on the same external drive as cache_root")
    if PROJECT_ROOT == destination or PROJECT_ROOT in destination.parents:
        raise ValueError("Full Grad-CAM outputs must be outside the project repository")
    if any(part.lower() == "onedrive" for part in destination.parts):
        raise ValueError("Full Grad-CAM outputs must be outside OneDrive")
    destination.mkdir(parents=True, exist_ok=True)
    subset = dataset.frame.reset_index(drop=True).set_index("image_id")
    report_records: list[dict[str, Any]] = []
    readme_examples_written: set[str] = set()

    with GradCAM(model, model.layer4[-1]) as gradcam:
        for item in examples:
            cached_path = config["cache_root"] / f"{item['image_id']}.jpg"
            image_tensor, _ = dataset[subset.index.get_loc(item["image_id"])]
            heatmaps, logits, targets = gradcam(image_tensor.unsqueeze(0), item["pred_idx"])
            if tuple(heatmaps.shape[-2:]) != (config["image_size"], config["image_size"]):
                raise AssertionError("Grad-CAM heatmap size does not match the configured input size")
            if int(targets[0]) != item["pred_idx"]:
                raise AssertionError("Grad-CAM target class changed unexpectedly")

            kind_dir = destination / item["kind"]
            kind_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{item['image_id']}_true-{CLASSES[item['true_idx']]}_pred-{CLASSES[item['pred_idx']]}.png"
            panel_path = kind_dir / file_name
            item["logit_target"] = float(logits[0, item["pred_idx"]])
            _write_panel(cached_path, heatmaps[0].cpu().numpy(), item, panel_path, include_three_panels=True)
            record = {
                "image_id": item["image_id"],
                "split": "validation",
                "kind": item["kind"],
                "true_class": CLASSES[item["true_idx"]],
                "predicted_class": CLASSES[item["pred_idx"]],
                "correct": item["correct"],
                "p_mel": item["p_mel"],
                "decision_threshold": threshold,
                "gradcam_target": CLASSES[item["pred_idx"]],
                "panel_path": str(panel_path),
            }
            report_records.append(record)

            # Keep just two compact, attributed README illustrations in-repo.
            example_name = "correct_melanoma.jpg" if item["correct"] else "missed_melanoma.jpg"
            if (
                include_readme_examples
                and item["true_idx"] == MEL_IDX
                and example_name not in readme_examples_written
            ):
                example_path = PROJECT_ROOT / "readme_images" / example_name
                _write_panel(
                    cached_path,
                    heatmaps[0].cpu().numpy(),
                    item,
                    example_path,
                    include_three_panels=False,
                )
                readme_examples_written.add(example_name)

    (destination / "examples.json").write_text(
        json.dumps(
            {
                "source_split": "validation only",
                "checkpoint": str(checkpoint_path),
                "validation_selected_melanoma_threshold": threshold,
                "target_layer": "layer4[-1]",
                "note": "Grad-CAM shows model attention, not medical correctness.",
                "examples": report_records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    _write_contact_sheet(report_records, destination / "contact_sheet.png")
    return report_records


def _write_contact_sheet(records: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(15, 6), dpi=120)
    for ax, record in zip(axes.flat, records):
        with Image.open(record["panel_path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"{record['image_id']}\n{record['true_class']}→{record['predicted_class']}",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle("Validation Grad-CAM: 6 correct, 6 incorrect", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to dataset_config.json")
    parser.add_argument("--output-dir", default=None, help="External folder for generated panels")
    parser.add_argument(
        "--include-readme-examples",
        action="store_true",
        help="Write two compact validation overlays into readme_images/",
    )
    args = parser.parse_args()
    records = generate_visualizations(args.config, args.output_dir, args.include_readme_examples)
    counts = pd.Series([item["kind"] for item in records]).value_counts()
    print(f"correct={int(counts.get('correct', 0))}; incorrect={int(counts.get('incorrect', 0))}")
    print(f"missed melanomas included={sum(item['true_class'] == 'mel' and not item['correct'] for item in records)}")
    for item in records:
        print(
            f"{item['kind']}: {item['image_id']} true={item['true_class']} "
            f"pred={item['predicted_class']} p_mel={item['p_mel']:.8f}"
        )


if __name__ == "__main__":
    main()
