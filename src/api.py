"""Local FastAPI service for the frozen DermaLens Stage B checkpoint."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.models import resnet18

from src.data_pipeline import CLASSES, PROJECT_ROOT, load_config, preprocess_image
from src.gradcam import GradCAM


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
IMAGE_SIZE = 224
DISCLAIMER = "Educational research prototype. Not a medical device."
MIME_FORMATS = {
    "image/jpeg": {"JPEG"},
    "image/png": {"PNG"},
    "image/webp": {"WEBP"},
}
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
_JET = matplotlib.colormaps["jet"]


def resolve_checkpoint_path(config: dict[str, Any]) -> Path:
    """Get Stage B's checkpoint path from the dataset config or its root."""
    configured = config.get("stage_b_checkpoint")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = config["checkpoint_root"] / path
        return path.resolve()
    return (config["checkpoint_root"] / "stage_b" / "best.pt").resolve()


def load_stage_b_model(checkpoint_path: Path) -> nn.Module:
    """Load exactly one frozen Stage B ResNet18 from the configured checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage B checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def decode_uploaded_image(file: UploadFile) -> Image.Image:
    """Validate MIME, byte size, actual image format, and image readability."""
    mime_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if mime_type not in MIME_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG, PNG, or WebP image.",
        )
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MiB upload limit (5,242,880 bytes).",
        )
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                image_format = probe.format
                probe.verify()
            if image_format not in MIME_FORMATS[mime_type]:
                raise HTTPException(
                    status_code=415,
                    detail="File content does not match its declared image type.",
                )
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is not a readable JPEG, PNG, or WebP image.",
        ) from exc


def preprocess_upload_image(
    image: Image.Image, image_size: int = IMAGE_SIZE
) -> tuple[Image.Image, torch.Tensor]:
    """Match cache creation's resize/JPEG round-trip and evaluation normalization."""
    resized = image.convert("RGB")
    resized = resized.resize((image_size, image_size), Image.Resampling.BILINEAR)
    encoded = io.BytesIO()
    resized.save(encoded, format="JPEG", quality=95, subsampling=0)
    encoded.seek(0)
    with Image.open(encoded) as cached_like_image:
        resized = cached_like_image.convert("RGB")
    tensor = preprocess_image(resized, image_size=image_size).unsqueeze(0)
    return resized, tensor


class InferenceService:
    """Thread-safe prediction and Grad-CAM service over one loaded model."""

    def __init__(self, model: nn.Module, image_size: int = IMAGE_SIZE):
        self.model = model.eval()
        self.image_size = image_size
        self.gradcam = GradCAM(self.model, self.model.layer4[-1])
        self._inference_lock = threading.Lock()

    def close(self) -> None:
        self.gradcam.close()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        # The shared Grad-CAM hook stores one activation tensor, so serialize
        # requests while using the singleton model and its hook.
        with self._inference_lock:
            started = time.perf_counter()
            resized, image_tensor = preprocess_upload_image(image, self.image_size)
            with torch.inference_mode():
                logits = self.model(image_tensor)
                probabilities_tensor = torch.softmax(logits, dim=1)
            predicted_index = int(probabilities_tensor.argmax(dim=1).item())
            heatmaps, _gradcam_logits, gradcam_targets = self.gradcam(
                image_tensor, target_classes=predicted_index
            )
            if int(gradcam_targets[0]) != predicted_index:
                raise RuntimeError("Grad-CAM target did not match the argmax prediction")
            overlay_base64 = _encode_overlay(resized, heatmaps[0].cpu().numpy())
            inference_time_ms = (time.perf_counter() - started) * 1000.0

        probabilities = probabilities_tensor[0].cpu().tolist()
        predicted_class = CLASSES[predicted_index]
        return {
            "predicted_class": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "class_probabilities": {
                class_name: float(probabilities[index])
                for index, class_name in enumerate(CLASSES)
            },
            "confidence": float(probabilities[predicted_index]),
            "gradcam_overlay_base64": overlay_base64,
            "gradcam_overlay_media_type": "image/jpeg",
            "inference_time_ms": inference_time_ms,
            "disclaimer": DISCLAIMER,
        }


def _encode_overlay(image: Image.Image, heatmap: np.ndarray) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    colored = _JET(heatmap)[..., :3]
    overlay = np.clip(0.55 * rgb + 0.45 * colored, 0.0, 1.0)
    output = io.BytesIO()
    Image.fromarray(np.uint8(np.round(overlay * 255.0))).save(
        output, format="JPEG", quality=85, optimize=True
    )
    return base64.b64encode(output.getvalue()).decode("ascii")


def _cors_origins() -> list[str]:
    configured = os.getenv("DERMALENS_CORS_ORIGINS")
    if not configured:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def create_app(
    config_path: str | Path | None = None,
    model_loader: Callable[[Path], nn.Module] | None = None,
) -> FastAPI:
    """Create an API app; optional loader injection supports isolated API tests."""
    configured_path = config_path or os.getenv("DERMALENS_CONFIG")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = None
        application.state.load_error = None
        try:
            config = load_config(configured_path)
            checkpoint_path = resolve_checkpoint_path(config)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Stage B checkpoint not found: {checkpoint_path}")
            torch.set_num_threads(int(config.get("inference_threads", 10)))
            loader = model_loader or load_stage_b_model
            model = loader(checkpoint_path)
            model.eval()
            application.state.service = InferenceService(
                model, image_size=int(config.get("image_size", IMAGE_SIZE))
            )
        except Exception as exc:  # health stays available and reports the load failure
            application.state.load_error = str(exc)
        try:
            yield
        finally:
            service = application.state.service
            if service is not None:
                service.close()

    application = FastAPI(
        title="DermaLens API",
        description="Educational image classification and Grad-CAM research prototype.",
        lifespan=lifespan,
    )
    from fastapi.middleware.cors import CORSMiddleware

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.exception_handler(HTTPException)
    async def http_error_handler(_request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "disclaimer": DISCLAIMER},
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "disclaimer": DISCLAIMER},
        )

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        loaded = application.state.service is not None
        result: dict[str, Any] = {
            "status": "ok" if loaded else "degraded",
            "backend_status": "available",
            "model_loaded": loaded,
            "disclaimer": DISCLAIMER,
        }
        if not loaded:
            result["model_load_error"] = application.state.load_error or "Model is not loaded."
        return result

    @application.get("/api/metrics")
    def metrics() -> FileResponse:
        metrics_path = METRICS_DIR / "metrics.json"
        if not metrics_path.is_file():
            raise HTTPException(status_code=404, detail="Metrics report is unavailable.")
        return FileResponse(metrics_path, media_type="application/json")

    @application.get("/api/metrics/confusion-matrix")
    def confusion_matrix_image() -> FileResponse:
        image_path = METRICS_DIR / "confusion_matrix.png"
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="Confusion matrix image is unavailable.")
        return FileResponse(image_path, media_type="image/png")

    @application.post("/api/predict")
    def predict(file: UploadFile = File(...)) -> dict[str, Any]:
        service = application.state.service
        if service is None:
            reason = application.state.load_error or "Model is not loaded."
            raise HTTPException(status_code=503, detail=f"Prediction unavailable: {reason}")
        image = decode_uploaded_image(file)
        try:
            return service.predict(image)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Inference failed; no prediction was returned.",
            ) from exc

    return application


app = create_app()
