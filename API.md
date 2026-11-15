# Local FastAPI backend

The backend serves the frozen Stage B ResNet18 checkpoint and a Grad-CAM
overlay. It loads the model once during startup. Configure
`stage_b_checkpoint` in the ignored `dataset_config.json`; when that optional
field is absent, the path is resolved from the configured
`checkpoint_root/stage_b/best.pt`. A missing checkpoint leaves the backend
healthy but degraded: `/api/health` reports `model_loaded: false`, while
`/api/predict` returns HTTP 503 with the missing path. The service never
returns a placeholder prediction.

Start the development server from the project directory:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

`POST /api/predict` accepts one multipart field named `file`. JPEG, PNG, and
WebP are accepted up to 5 MiB (5,242,880 bytes). The API verifies the declared MIME type, file
contents, and image readability, then matches cache preprocessing: 224×224
bilinear resize, JPEG quality 95 with no chroma subsampling, followed by the
same ImageNet tensor conversion and normalization used by evaluation. The response
contains the argmax class, readable class name, all seven probabilities,
confidence, an `image/jpeg` Grad-CAM overlay as base64, inference time in
milliseconds, and the disclaimer:

> Educational research prototype. Not a medical device.

The classifier uses argmax over the seven class probabilities. It does not use
the validation-selected melanoma threshold. Inference time covers preprocessing
of the decoded image, prediction, Grad-CAM, and overlay encoding; file upload
transfer and JSON transmission are outside that timer.

`GET /api/health` reports backend status, model-loaded status, and any model
load error. JSON health, validation, and prediction error responses also carry
the disclaimer. CORS is enabled by default for React development servers on
`localhost` and `127.0.0.1` ports 3000 and 5173. Override the list with the
comma-separated `DERMALENS_CORS_ORIGINS` environment variable.

Run API and preprocessing tests with:

```powershell
python -m pytest tests/test_api.py -q
```

Measure CPU prediction plus Grad-CAM latency with one warm-up and 20 measured
requests. The script uses the first cached validation image and does not access
the test split or write metrics:

```powershell
python -m src.benchmark_api --requests 20 --warmup 1
```
