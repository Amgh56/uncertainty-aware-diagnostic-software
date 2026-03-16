"""
ML state holder and dynamic model loader.

- Legacy model: loaded once at startup (the hardcoded CheXpert model)
- Published models: loaded on-demand with LRU cache (bounded)
"""

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from conformal_prediction_pipeline import (
    DISEASES,
    load_chexpert_pretrained_model,
    pick_device,
    predict_batch_probs,
    preprocess_image_from_bytes,
)
from supabase_client import upload_image

BACKEND_DIR = Path(__file__).resolve().parent.parent  # services/ -> backend/

MAX_CACHED_MODELS = 5


@dataclass
class LoadedModel:
    """A loaded model with all its inference parameters."""
    model: object
    config: Optional[dict]
    device: str
    lamhat: float
    alpha: float
    labels: list[str]
    artifact_type: str = "pytorch"


@dataclass
class MLState:
    """Holds the legacy model and a cache of published models."""
    model: Optional[object] = None
    config: Optional[object] = None
    device: Optional[str] = None
    lamhat: Optional[float] = None
    alpha: Optional[float] = None
    diseases: list = field(default_factory=lambda: list(DISEASES))

    # LRU cache for published models: model_id -> LoadedModel
    _model_cache: OrderedDict = field(default_factory=OrderedDict)

    def load(self) -> None:
        """Load the legacy hardcoded model at startup."""
        self.model, self.config = load_chexpert_pretrained_model()
        self.device = pick_device()
        self.model = self.model.to(self.device)
        self.lamhat, self.alpha = self._load_lamhat_json()
        print(f"Model loaded on device: {self.device}")
        print(f"Loaded lamhat={self.lamhat:.6f}, alpha={self.alpha}")

    def _load_lamhat_json(self) -> tuple[float, float]:
        lamhat_path = BACKEND_DIR / "NIH_dataset" / "artifacts" / "lamhat.json"
        if not lamhat_path.exists():
            raise FileNotFoundError(
                f"lamhat.json not found at: {lamhat_path}. Run calibration first."
            )
        with open(lamhat_path, "r") as f:
            payload = json.load(f)
        if "lamhat" not in payload or "alpha" not in payload:
            raise ValueError(f"lamhat payload missing keys in {lamhat_path}")
        return float(payload["lamhat"]), float(payload["alpha"])

    # ── Legacy inference (no model_id) ────────────────────────

    def run_inference(self, img_bytes: bytes) -> list[dict]:
        """Legacy: run using the hardcoded CheXpert model."""
        img_array = preprocess_image_from_bytes(img_bytes, self.config)
        x = torch.tensor(
            img_array[np.newaxis], dtype=torch.float32, device=self.device
        )
        probs = predict_batch_probs(self.model, x)[0]

        findings = []
        for i, disease in enumerate(DISEASES):
            p = float(probs[i])
            in_set = bool(p >= self.lamhat)
            uncertainty = _classify_uncertainty(p)
            findings.append({
                "finding": disease,
                "probability": round(p, 4),
                "uncertainty": uncertainty,
                "in_prediction_set": in_set,
            })

        findings.sort(key=lambda row: row["probability"], reverse=True)
        return findings

    # ── Dynamic inference (with published model) ──────────────

    def load_published_model(self, published_model) -> LoadedModel:
        """Load a published model, using cache if available."""
        model_id = published_model.id

        # Cache hit — move to end (most recently used)
        if model_id in self._model_cache:
            self._model_cache.move_to_end(model_id)
            return self._model_cache[model_id]

        # Cache miss — load from disk
        artifact_path = Path(published_model.artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}"
            )

        device = pick_device()

        # Load model
        model_obj = _load_model_artifact(artifact_path, device)

        # Parse config
        config_dict = None
        if published_model.config_json:
            config_dict = json.loads(published_model.config_json)

        # Parse labels
        labels = json.loads(published_model.labels_json)

        loaded = LoadedModel(
            model=model_obj,
            config=config_dict,
            device=device,
            lamhat=published_model.lamhat,
            alpha=published_model.alpha,
            labels=labels,
            artifact_type=published_model.artifact_type,
        )

        # Evict LRU if at capacity
        if len(self._model_cache) >= MAX_CACHED_MODELS:
            self._model_cache.popitem(last=False)

        self._model_cache[model_id] = loaded
        print(f"Loaded published model {model_id[:8]} ({len(labels)} labels) on {device}")
        return loaded

    def run_published_inference(
        self, img_bytes: bytes, published_model
    ) -> list[dict]:
        """Run inference using a published model's artifact, config, lamhat, and labels."""
        loaded = self.load_published_model(published_model)

        # Preprocess image using the model's config
        img_array = _preprocess_with_config(img_bytes, loaded.config)
        x = torch.tensor(
            img_array[np.newaxis], dtype=torch.float32, device=loaded.device
        )

        # Forward pass
        probs = _forward(loaded.model, x)

        # Build findings using the model's labels and lamhat
        findings = []
        for i, label in enumerate(loaded.labels):
            p = float(probs[i])
            in_set = bool(p >= loaded.lamhat)
            uncertainty = _classify_uncertainty(p)
            findings.append({
                "finding": label,
                "probability": round(p, 4),
                "uncertainty": uncertainty,
                "in_prediction_set": in_set,
            })

        findings.sort(key=lambda row: row["probability"], reverse=True)
        return findings

    # ── Image upload ──────────────────────────────────────────

    def upload_xray(
        self, img_bytes: bytes, doctor_id: int, filename: str, content_type: str
    ) -> str:
        """Upload image to Supabase Storage, return public URL."""
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        storage_path = f"{doctor_id}/{safe_filename}"
        return upload_image(img_bytes, storage_path, content_type)


# ── Helpers ──────────────────────────────────────────────────

def _classify_uncertainty(p: float) -> str:
    if p >= 0.7:
        return "Low"
    elif p >= 0.4:
        return "Medium"
    return "High"


def _load_model_artifact(path: Path, device: str) -> torch.nn.Module:
    """Load a model from disk (TorchScript or full saved model)."""
    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    raise ValueError(f"Cannot load model from {path}")


def _preprocess_with_config(img_bytes: bytes, config: dict | None) -> np.ndarray:
    """Preprocess image bytes using model-specific config."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")

    img = img.astype(np.float32)

    if config:
        w = int(config.get("width", 224))
        h = int(config.get("height", 224))
        img = cv2.resize(img, (w, h))
        if config.get("use_equalizeHist", False):
            img = cv2.equalizeHist(img.astype(np.uint8)).astype(np.float32)
        mean = float(config.get("pixel_mean", 128.0))
        std = float(config.get("pixel_std", 64.0))
        img = (img - mean) / std
    else:
        # Default: 224x224, standard normalization
        img = cv2.resize(img, (224, 224))
        img = (img - 128.0) / 64.0

    return np.stack([img, img, img], axis=0)


def _forward(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Run inference and return sigmoid probabilities as numpy (n_classes,)."""
    model.eval()
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            logits = out[0]
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        else:
            logits = out
        return torch.sigmoid(logits).cpu().numpy()[0]


# Module-level singleton
ml_state = MLState()
