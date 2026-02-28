"""
Singleton ML state holder. Loaded once at server startup.

Encapsulates model loading, inference, conformal prediction logic,
and image upload — keeping these details out of route handlers.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass
class MLState:
    model: Optional[object] = None
    config: Optional[object] = None
    device: Optional[str] = None
    lamhat: Optional[float] = None
    alpha: Optional[float] = None
    diseases: list = field(default_factory=lambda: list(DISEASES))

    def load(self) -> None:
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

    def run_inference(self, img_bytes: bytes) -> list[dict]:
        """Preprocess image, run model, apply conformal thresholding, return findings."""
        img_array = preprocess_image_from_bytes(img_bytes, self.config)
        x = torch.tensor(
            img_array[np.newaxis], dtype=torch.float32, device=self.device
        )
        probs = predict_batch_probs(self.model, x)[0]

        findings = []
        for i, disease in enumerate(DISEASES):
            p = float(probs[i])
            in_set = bool(p >= self.lamhat)

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
        return findings

    def upload_xray(
        self, img_bytes: bytes, doctor_id: int, filename: str, content_type: str
    ) -> str:
        """Upload image to Supabase Storage, return public URL."""
        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{filename}"
        storage_path = f"{doctor_id}/{safe_filename}"
        return upload_image(img_bytes, storage_path, content_type)


# Module-level singleton
ml_state = MLState()
