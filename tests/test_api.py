import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from src.api import DISCLAIMER, MAX_UPLOAD_BYTES, create_app, preprocess_upload_image
from src.data_pipeline import CLASSES, DermImageDataset, cache_resized_images


class TinyResNet(nn.Module):
    """Small test model with the layer4/fc interface Grad-CAM expects."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 7)

    def forward(self, images):
        features = self.layer4(self.stem(images))
        return self.fc(torch.flatten(self.pool(features), 1))


def _write_config(root, checkpoint):
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "dataset_config.json"
    config = {
        "dataset_root": str(root / "data"),
        "cache_root": str(root / "cache"),
        "checkpoint_root": str(root / "checkpoints"),
        "stage_b_checkpoint": str(checkpoint),
        "metadata_file": "metadata.csv",
        "image_directories": ["images"],
        "image_size": 224,
        "seed": 42,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _jpeg_bytes(image: Image.Image | None = None) -> bytes:
    if image is None:
        rng = np.random.default_rng(42)
        image = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


@pytest.fixture(scope="module")
def healthy_client(tmp_path_factory):
    root = tmp_path_factory.mktemp("dermalens-api")
    checkpoint = root / "stage_b_best.pt"
    checkpoint.write_bytes(b"test checkpoint marker")
    config_path = _write_config(root, checkpoint)
    load_calls = []

    def model_loader(path):
        load_calls.append(path)
        torch.manual_seed(42)
        return TinyResNet()

    app = create_app(config_path=config_path, model_loader=model_loader)
    with TestClient(app) as client:
        yield client, load_calls


def test_predict_happy_path_returns_prediction_probabilities_and_gradcam(healthy_client):
    client, load_calls = healthy_client
    response = client.post(
        "/api/predict",
        files={"file": ("lesion.jpg", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASSES
    assert body["predicted_class_name"]
    assert set(body["class_probabilities"]) == set(CLASSES)
    assert sum(body["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["predicted_class"] == max(
        body["class_probabilities"], key=body["class_probabilities"].get
    )
    assert body["confidence"] == pytest.approx(
        max(body["class_probabilities"].values()), abs=1e-7
    )
    assert body["gradcam_overlay_media_type"] == "image/jpeg"
    assert len(base64.b64decode(body["gradcam_overlay_base64"], validate=True)) > 100
    assert body["inference_time_ms"] >= 0
    assert body["disclaimer"] == DISCLAIMER
    assert len(load_calls) == 1


def test_predict_rejects_invalid_file_type(healthy_client):
    client, _ = healthy_client
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
    assert "JPEG, PNG, or WebP" in response.json()["detail"]
    assert response.json()["disclaimer"] == DISCLAIMER


def test_predict_rejects_corrupt_image(healthy_client):
    client, _ = healthy_client
    response = client.post(
        "/api/predict",
        files={"file": ("broken.jpg", b"not a valid jpeg", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "readable" in response.json()["detail"]
    assert response.json()["disclaimer"] == DISCLAIMER


def test_predict_rejects_oversized_file(healthy_client):
    client, _ = healthy_client
    response = client.post(
        "/api/predict",
        files={"file": ("large.jpg", b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")},
    )
    assert response.status_code == 413
    assert "5 MiB" in response.json()["detail"]
    assert response.json()["disclaimer"] == DISCLAIMER


def test_health_reports_backend_and_loaded_model(healthy_client):
    client, _ = healthy_client
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "backend_status": "available",
        "model_loaded": True,
        "disclaimer": DISCLAIMER,
    }


def test_cors_allows_local_react_dev_server(healthy_client):
    client, _ = healthy_client
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_metrics_endpoint_returns_committed_metrics_json(healthy_client):
    client, _ = healthy_client
    response = client.get("/api/metrics")
    expected = json.loads(
        (Path(__file__).resolve().parents[1] / "artifacts" / "metrics" / "metrics.json")
        .read_text(encoding="utf-8")
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_confusion_matrix_endpoint_serves_metrics_image(healthy_client):
    client, _ = healthy_client
    response = client.get("/api/metrics/confusion-matrix")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_missing_checkpoint_returns_clear_unavailable_response(tmp_path):
    missing_checkpoint = tmp_path / "missing" / "best.pt"
    config_path = _write_config(tmp_path, missing_checkpoint)
    app = create_app(config_path=config_path)
    with TestClient(app) as client:
        health = client.get("/api/health")
        prediction = client.post(
            "/api/predict",
            files={"file": ("lesion.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert str(missing_checkpoint) in health.json()["model_load_error"]
    assert prediction.status_code == 503
    assert "Stage B checkpoint not found" in prediction.json()["detail"]
    assert prediction.json()["disclaimer"] == DISCLAIMER


def test_api_preprocessing_matches_evaluation_pipeline(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    cache.mkdir()
    pixels = np.arange(271 * 319 * 3, dtype=np.uint32).reshape(271, 319, 3)
    image = Image.fromarray((pixels % 256).astype(np.uint8))
    source_path = source / "same_image.png"
    image.save(source_path, format="PNG")
    cache_resized_images(
        {"same_image": source_path}, cache, dataset_root=source, image_size=224
    )
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        [{"image_id": "same_image", "lesion_id": "lesion-1", "dx": "nv"}]
    ).to_csv(split_path, index=False)
    evaluation_dataset = DermImageDataset(split_path, cache, augment=False)

    with Image.open(source_path) as uploaded_image:
        _resized, api_tensor = preprocess_upload_image(uploaded_image)
    evaluation_tensor, _label = evaluation_dataset[0]

    torch.testing.assert_close(api_tensor[0], evaluation_tensor, rtol=0, atol=0)
