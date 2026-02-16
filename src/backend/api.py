"""
FastAPI backend for the Uncertainty-Aware Diagnostic System.

Runs single-image conformal prediction inference using the pretrained CheXpert model
and calibrated lamhat loaded once at server startup.

Usage:
    cd src/backend
    pip install -r requirements-api.txt
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
import json
import os

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from conformal_prediction_pipeline import (
    DISEASES,
    load_chexpert_pretrained_model,
    pick_device,
    predict_batch_probs,
    preprocess_image_from_bytes,
)


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent


app = FastAPI(
    title="Uncertainty-Aware Chest X-ray Diagnostic API",
    description="Conformal prediction sets for multi-label chest X-ray classification",
    version="1.0.0",
)

cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
if cors_origins_raw.strip() == "*":
    cors_origins = ["*"]
    allow_credentials = False
else:
    cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
config = None
device = None
lamhat = None
alpha = None


def is_supported_upload(file: UploadFile) -> bool:
    if not file:
        return False

    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    if content_type in {"image/png", "image/jpeg"}:
        return True

    if filename.endswith((".png", ".jpg", ".jpeg")):
        return True

    return False


def load_lamhat_json() -> tuple[float, float, Path]:
    lamhat_path = BACKEND_DIR / "NIH_dataset" / "artifacts" / "lamhat.json"

    if not lamhat_path.exists():
        raise FileNotFoundError(
            f"lamhat.json not found at: {lamhat_path}. Run calibration first."
        )

    with open(lamhat_path, "r") as f:
        payload = json.load(f)

    if "lamhat" not in payload or "alpha" not in payload:
        raise ValueError(f"lamhat payload missing keys in {lamhat_path}")

    return float(payload["lamhat"]), float(payload["alpha"]), lamhat_path


@app.on_event("startup")
def startup() -> None:
    global model, config, device, lamhat, alpha

    model, config = load_chexpert_pretrained_model()
    device = pick_device()
    model = model.to(device)

    lamhat, alpha, lamhat_path = load_lamhat_json()

    print(f"Model loaded on device: {device}")
    print(f"Loaded lamhat={lamhat:.6f}, alpha={alpha} from {lamhat_path}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not is_supported_upload(file):
        raise HTTPException(status_code=400, detail="File must be PNG, JPEG")

    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise ValueError("Uploaded file is empty")

        img_array = preprocess_image_from_bytes(img_bytes, config)

        x = torch.tensor(
            img_array[np.newaxis],
            dtype=torch.float32,
            device=device,
        )

        probs = predict_batch_probs(model, x)[0]

        findings = []
        for i, disease in enumerate(DISEASES):
            p = float(probs[i])
            in_set = bool(p >= lamhat)

            if p >= 0.7:
                uncertainty = "Low"
            elif p >= 0.4:
                uncertainty = "Medium"
            else:
                uncertainty = "High"

            findings.append(
                {
                    "finding": disease,
                    "probability": round(p, 4),
                    "uncertainty": uncertainty,
                    "in_prediction_set": in_set,
                }
            )

        findings.sort(key=lambda row: row["probability"], reverse=True)

        prediction_set_size = sum(1 for row in findings if row["in_prediction_set"])
        top = findings[0]

        return {
            "top_finding": top["finding"],
            "top_probability": top["probability"],
            "prediction_set_size": prediction_set_size,
            "coverage": f"{int((1 - alpha) * 100)}%",
            "alpha": alpha,
            "lamhat": round(lamhat, 6),
            "findings": findings,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "lamhat": lamhat,
        "alpha": alpha,
        "diseases": DISEASES,
    }
