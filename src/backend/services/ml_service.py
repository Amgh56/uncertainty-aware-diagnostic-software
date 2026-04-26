import io
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

from services.calibration_service import pick_device
from azure_client import BUCKET_MODELS, download_from_bucket, upload_image
from enums import ArtifactType, UncertaintyLevel

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
    artifact_type: str = ArtifactType.PYTORCH.value


@dataclass
class MLState:
    """Holds a cache of published models loaded from Supabase."""

    _model_cache: OrderedDict = field(default_factory=OrderedDict)

    def load_published_model(self, published_model) -> LoadedModel:
        """Load a published model, using cache if available."""
        model_id = published_model.id

        if model_id in self._model_cache:
            self._model_cache.move_to_end(model_id)
            return self._model_cache[model_id]

        # Cache miss — download from Supabase
        try:
            model_bytes = download_from_bucket(BUCKET_MODELS, published_model.artifact_path)
        except Exception:
            raise FileNotFoundError(
                f"Model artifact not found in storage: {published_model.artifact_path}"
            )

        device = pick_device()

        model_obj = load_model_from_bytes(model_bytes, device)

        config_dict = None
        if published_model.config_json:
            config_dict = json.loads(published_model.config_json)

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

        if len(self._model_cache) >= MAX_CACHED_MODELS:
            self._model_cache.popitem(last=False)

        self._model_cache[model_id] = loaded
        print(f"Loaded published model {model_id[:8]} ({len(labels)} labels) on {device}")
        return loaded

    def run_inference(
        self, img_bytes: bytes, published_model
    ) -> list[dict]:
        """Run inference using a published model's artifact, config, lamhat, and labels."""
        loaded = self.load_published_model(published_model)

        # Preprocess image using the model's config
        img_array = preprocess_with_config(img_bytes, loaded.config)
        x = torch.tensor(
            img_array[np.newaxis], dtype=torch.float32, device=loaded.device
        )

        # Forward pass
        probs = forward(loaded.model, x)

        # Build findings using the model's labels and lamhat
        findings = []
        for i, label in enumerate(loaded.labels):
            p = float(probs[i])
            in_set = bool(p >= loaded.lamhat)
            uncertainty = classify_uncertainty(p)
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
        self, img_bytes: bytes, user_id: int, filename: str, content_type: str
    ) -> str:
        """Upload image to Supabase Storage, return public URL."""
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        storage_path = f"{user_id}/{safe_filename}"
        return upload_image(img_bytes, storage_path, content_type)


# ── Helpers ──────────────────────────────────────────────────

def classify_uncertainty(p: float) -> str:
    if p >= 0.7:
        return UncertaintyLevel.LOW.value
    elif p >= 0.4:
        return UncertaintyLevel.MEDIUM.value
    return UncertaintyLevel.HIGH.value


def load_model_from_bytes(model_bytes: bytes, device: str) -> torch.nn.Module:
    """Load a model from bytes (TorchScript or full saved model)."""
    buffer = io.BytesIO(model_bytes)

    try:
        buffer.seek(0)
        model = torch.jit.load(buffer, map_location=device)
        model.eval()
        return model
    except Exception:
        pass

    buffer.seek(0)
    obj = torch.load(buffer, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    raise ValueError("Cannot load model from bytes")


def preprocess_with_config(img_bytes: bytes, config: dict | None) -> np.ndarray:
    """Preprocess image bytes → float32 tensor (3, H, W) ready for any PyTorch model.

    Modality is auto-detected from the image data — no config field required:

      Grayscale  (X-ray, CT, MRI, ultrasound, …)
        All three BGR channels are identical → treat as single-channel.
        Normalised with pixel_mean / pixel_std (defaults: 128, 64).
        Channels are stacked 3× so the tensor shape matches RGB-pretrained models.

      Colour  (retinal fundus, dermoscopy, pathology slides, …)
        BGR channels differ → treat as colour.
        Converted to RGB and normalised with ImageNet mean/std (0.485/0.456/0.406,
        0.229/0.224/0.225), which suits any ImageNet-pretrained backbone.

    Adding support for a new image type requires no code change — the correct
    path is selected automatically based on whether the image carries colour info.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")

    w = int(config.get("width",  224)) if config else 224
    h = int(config.get("height", 224)) if config else 224

    is_grayscale = np.array_equal(bgr[:, :, 0], bgr[:, :, 1]) and \
                   np.array_equal(bgr[:, :, 1], bgr[:, :, 2])

    if is_grayscale:
        gray = bgr[:, :, 0].astype(np.float32)
        gray = cv2.resize(gray, (w, h))
        if config and config.get("use_equalizeHist", False):
            gray = cv2.equalizeHist(gray.astype(np.uint8)).astype(np.float32)
        mean = float(config.get("pixel_mean", 128.0)) if config else 128.0
        std  = float(config.get("pixel_std",  64.0))  if config else 64.0
        gray = (gray - mean) / std
        return np.stack([gray, gray, gray], axis=0)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (w, h))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        return rgb.transpose(2, 0, 1)  # (H, W, 3) → (3, H, W)


def forward(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
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


ml_state = MLState()
