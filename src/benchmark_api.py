"""Measure local CPU API latency using warm-up plus repeated validation requests."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.api import create_app, resolve_checkpoint_path
from src.data_pipeline import SPLIT_DIR, load_config


def benchmark(config_path: str | Path | None, request_count: int, warmup_count: int) -> dict:
    config = load_config(config_path)
    validation = pd.read_csv(SPLIT_DIR / "val.csv")
    if validation.empty:
        raise RuntimeError("Validation split is empty; cannot run the API latency benchmark")
    image_id = str(validation.iloc[0]["image_id"])
    image_path = config["cache_root"] / f"{image_id}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"Validation cache image is missing: {image_path}")
    image_bytes = image_path.read_bytes()
    app = create_app(config_path=config_path)
    inference_times: list[float] = []
    request_times: list[float] = []

    with TestClient(app) as client:
        health = client.get("/api/health")
        if health.status_code != 200 or not health.json().get("model_loaded"):
            raise RuntimeError(f"API model did not load: {health.json()}")
        total_requests = warmup_count + request_count
        for index in range(total_requests):
            started = time.perf_counter()
            response = client.post(
                "/api/predict",
                files={"file": (f"{image_id}.jpg", image_bytes, "image/jpeg")},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.status_code != 200:
                raise RuntimeError(f"API request failed: {response.status_code} {response.text}")
            body = response.json()
            if index >= warmup_count:
                inference_times.append(float(body["inference_time_ms"]))
                request_times.append(elapsed_ms)
            if (index + 1) % 5 == 0 or index + 1 == total_requests:
                print(f"benchmark requests completed: {index + 1}/{total_requests}", flush=True)

    return {
        "device": "cpu",
        "checkpoint": str(resolve_checkpoint_path(config)),
        "source_split": "validation",
        "validation_image_id": image_id,
        "warmup_requests": warmup_count,
        "measured_requests": request_count,
        "mean_inference_ms_prediction_plus_gradcam": statistics.mean(inference_times),
        "mean_http_round_trip_ms": statistics.mean(request_times),
        "inference_samples_ms": inference_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to dataset_config.json")
    parser.add_argument("--requests", type=int, default=20, help="Measured prediction requests")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up prediction requests")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("--requests must be positive and --warmup cannot be negative")
    result = benchmark(args.config, args.requests, args.warmup)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
