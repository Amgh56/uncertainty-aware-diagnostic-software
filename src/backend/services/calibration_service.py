"""
Calibration service for the Developer/Researcher mode.

Design principle: conformal_prediction_pipeline.py is kept unchanged (CheXpert-specific).
All model-agnostic logic lives here as local functions so any model + any labelled dataset works.

Handles:
- Generic model loading  (TorchScript or full saved nn.Module)
- Preprocessing driven by the uploaded config.json (width, height, pixel_mean, pixel_std)
- Generic image preprocessing + inference  (rescales to match the model's expected dimensions)
- Dataset zip validation and extraction
- Auto-detecting labels from labels.csv (any columns, not just the 5 CheXpert diseases)
- Running conformal calibration pipeline
- Saving lamhat.json with metrics
- TTL-based cleanup of uploaded artefacts
"""

import json
import os
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

# Generic pipeline helpers — these don't depend on a specific model architecture.
from conformal_prediction_pipeline import (
    compute_lamhat,
    false_negative_rate,
    pick_device,
)
from database import SessionLocal
from models import CalibrationJob, JobStatus

BACKEND_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BACKEND_DIR / "developer_uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MIN_IMAGES = 50
MODEL_SIZE_LIMIT = 500 * 1024 * 1024        # 500 MB
DATASET_SIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GB
TTL_DAYS = int(os.getenv("DEVELOPER_JOB_TTL_DAYS", "7"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def job_dir(job_id: str) -> Path:
    return UPLOADS_DIR / job_id


def _read_preproc(config_path: Path) -> dict | None:
    """
    Read preprocessing parameters from the uploaded config.json.

    Required fields (when config is present): width, height, pixel_mean, pixel_std.
    Optional: use_equalizeHist (default False).

    Returns None if no config was uploaded — images will be used at their native size.
    """
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "width":            int(cfg["width"]),
        "height":           int(cfg["height"]),
        "pixel_mean":       float(cfg["pixel_mean"]),
        "pixel_std":        float(cfg["pixel_std"]),
        "use_equalizeHist": bool(cfg.get("use_equalizeHist", False)),
    }


# ---------------------------------------------------------------------------
# 1. Generic model loader
# ---------------------------------------------------------------------------

def load_generic_model(model_path: Path) -> torch.nn.Module:
    """
    Load a model file.  Two formats are accepted:

    1. **TorchScript** (recommended) — saved with::

           traced = torch.jit.trace(model, example_input)
           torch.jit.save(traced, "model.pt")

       Self-contained: no class imports needed at load time.
       Works with any architecture.

    2. **Full nn.Module** — saved with ``torch.save(model, path)``.
       The model class must be importable in this environment.

    State dicts (``OrderedDict``) are **not** accepted because they carry
    no architecture information.  The error message tells the uploader
    exactly how to convert one.
    """
    # --- Try TorchScript first (most general, architecture-agnostic) ---
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        pass  # not a TorchScript file — fall through

    # --- Try full saved nn.Module ---
    try:
        obj = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not read model file: {exc}") from exc

    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # --- Friendly error for state dicts / unknown formats ---
    raise ValueError(
        f"Unsupported model format (got {type(obj).__name__}). "
        f"Accepted formats:\n"
        f"  1. TorchScript (recommended):\n"
        f"       traced = torch.jit.trace(model, torch.zeros(1, 3, H, W))\n"
        f"       torch.jit.save(traced, 'model.pt')\n"
        f"  2. Full saved model:\n"
        f"       torch.save(model, 'model.pth')\n"
        f"State dicts (OrderedDict) are not accepted — they carry no "
        f"architecture information."
    )


# ---------------------------------------------------------------------------
# 2. Generic preprocessing + inference
# ---------------------------------------------------------------------------

def _preprocess_image(img_path: Path, preproc: dict | None) -> np.ndarray:
    """
    Read a grayscale image and return a float32 array (3, H, W).

    No config uploaded (preproc is None):
      - Images are read at their native size — no resizing, no normalisation.
      - The caller is responsible for ensuring all images are the same size.

    Config uploaded (preproc is a dict):
      - Resize to (width × height) from the config.
      - Normalise with pixel_mean / pixel_std from the config.
      - Optionally apply histogram equalisation if use_equalizeHist is True.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = img.astype(np.float32)

    if preproc is not None:
        img = cv2.resize(img, (preproc["width"], preproc["height"]))
        if preproc.get("use_equalizeHist", False):
            img = cv2.equalizeHist(img.astype(np.uint8)).astype(np.float32)
        img = (img - preproc["pixel_mean"]) / preproc["pixel_std"]

    return np.stack([img, img, img], axis=0)  # (3, H, W)


def _forward(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """
    Run inference and return sigmoid probabilities as numpy (B, n_classes).
    Handles both:
      - Standard output:  tensor of shape (B, n_classes)
      - CheXpert-style:   (list_or_tensor, _) tuple
    """
    model.eval()
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            logits = out[0]
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        else:
            logits = out
        return torch.sigmoid(logits).cpu().numpy()


def infer_all_probs_generic(
    model: torch.nn.Module,
    preproc: dict | None,
    img_paths: list,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Batched inference over all images.
    Each image is rescaled to (width × height) as specified in the uploaded config.json.
    Returns probabilities of shape (N, n_classes).
    """
    device = pick_device()
    model = model.to(device)

    all_probs = []
    for i in range(0, len(img_paths), batch_size):
        batch = [_preprocess_image(p, preproc) for p in img_paths[i:i + batch_size]]
        x = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
        all_probs.append(_forward(model, x))

    return np.concatenate(all_probs, axis=0)  # (N, n_classes)


# ---------------------------------------------------------------------------
# 3. Generic evaluation (label-agnostic)
# ---------------------------------------------------------------------------

def _evaluate_sets(
    pred_sets: np.ndarray,
    labels: np.ndarray,
    pos_mask: np.ndarray,
    label_names: list,
) -> dict:
    """
    Compute calibration metrics for any number of labels.
    Mirrors conformal_prediction_pipeline.evaluate_prediction_sets but is
    not tied to the global DISEASES constant or a fixed class count.
    """
    n_classes = len(label_names)
    set_sizes = pred_sets.sum(axis=1)

    results = {
        "fnr": false_negative_rate(pred_sets, labels, pos_mask),
        "avg_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "empty_set_rate": float((set_sizes == 0).mean()),
        "full_set_rate": float((set_sizes == n_classes).mean()),
        "size_distribution": {
            int(s): int((set_sizes == s).sum()) for s in range(n_classes + 1)
        },
    }

    per_disease = {}
    for j, name in enumerate(label_names):
        pos_j = labels[:, j] == 1
        n_pos = int(pos_j.sum())
        if n_pos > 0:
            recall_j = float(pred_sets[pos_j, j].mean())
            per_disease[name] = {
                "n_positive": n_pos,
                "recall": round(recall_j, 4),
                "fnr": round(1.0 - recall_j, 4),
                "pred_rate": round(float(pred_sets[:, j].mean()), 4),
            }
        else:
            per_disease[name] = {"n_positive": 0, "recall": None, "fnr": None}

    results["per_disease"] = per_disease
    return results


# ---------------------------------------------------------------------------
# 4. Zip validation + labels.csv parsing
# ---------------------------------------------------------------------------

def validate_and_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """
    Validate zip structure, protect against path traversal, extract to dest_dir.
    Returns the path to the extracted dataset root (contains images/ + labels.csv).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

        for member in members:
            if member.startswith("/") or ".." in member:
                raise ValueError(f"Unsafe path in zip: {member!r}")

        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        has_wrapper = (
            len(top_dirs) == 1
            and all(m.startswith(list(top_dirs)[0] + "/") for m in members if m != list(top_dirs)[0] + "/")
        )
        zf.extractall(dest_dir)

    dataset_root = dest_dir / list(top_dirs)[0] if has_wrapper else dest_dir

    if not (dataset_root / "images").is_dir():
        raise ValueError("Dataset zip must contain an 'images/' folder.")
    if not (dataset_root / "labels.csv").exists():
        raise ValueError("Dataset zip must contain a 'labels.csv' file.")

    return dataset_root


def validate_labels_csv(labels_csv: Path, images_dir: Path) -> tuple:
    """
    Parse labels.csv and auto-detect label columns.
    Any column other than 'filename' is treated as a label — no fixed disease list.
    Returns (img_paths, labels_array, pos_mask, label_names).
    """
    import pandas as pd

    df = pd.read_csv(labels_csv)

    if "filename" not in df.columns:
        raise ValueError("labels.csv must have a 'filename' column.")

    label_names = [c for c in df.columns if c != "filename"]
    if not label_names:
        raise ValueError("labels.csv must have at least one label column besides 'filename'.")

    if len(df) < MIN_IMAGES:
        raise ValueError(
            f"Dataset must have at least {MIN_IMAGES} labelled images (got {len(df)})."
        )

    img_paths = []
    for fname in df["filename"]:
        p = images_dir / fname
        if not p.exists():
            raise ValueError(f"Image referenced in labels.csv not found: {fname}")
        img_paths.append(p)

    labels = df[label_names].values.astype(np.float32)
    pos_mask = (labels.sum(axis=1) > 0)

    return img_paths, labels, pos_mask, label_names


# ---------------------------------------------------------------------------
# 5. Background calibration task
# ---------------------------------------------------------------------------

def run_calibration_job(job_id: str) -> None:
    """
    Run the full calibration pipeline for a job.
    Called as a FastAPI BackgroundTask — updates DB status directly.
    """
    db = SessionLocal()
    try:
        job = db.query(CalibrationJob).filter(CalibrationJob.id == job_id).first()
        if job is None:
            return

        job.status = JobStatus.RUNNING
        db.commit()

        base        = job_dir(job_id)
        model_path  = base / "model.pth"
        config_path = base / "config.json"
        dataset_zip = base / "dataset.zip"
        dataset_dir = base / "dataset"
        result_dir  = base / "result"
        result_dir.mkdir(exist_ok=True)
        dataset_dir.mkdir(exist_ok=True)

        # 1. Read preprocessing config from the uploaded config.json (optional).
        #    Returns None if no config was uploaded → images used at native size.
        preproc = _read_preproc(config_path)

        # 2. Extract + validate zip
        dataset_root = validate_and_extract_zip(dataset_zip, dataset_dir)

        # 3. Validate labels.csv — auto-detect any label columns
        images_dir = dataset_root / "images"
        labels_csv = dataset_root / "labels.csv"
        img_paths, labels, pos_mask, label_names = validate_labels_csv(labels_csv, images_dir)

        # 4. Load model — preprocessing is set by config.json, not the model file
        model = load_generic_model(model_path)
        model = model.to(pick_device())

        # 5. Inference — images rescaled to (width × height) from the uploaded config
        probs = infer_all_probs_generic(model, preproc, img_paths, batch_size=16)

        # 6. Conformal calibration
        lamhat = compute_lamhat(probs, labels, pos_mask, job.alpha)

        # 7. Evaluate
        pred_sets = (probs >= lamhat).astype(np.float32)
        metrics = _evaluate_sets(pred_sets, labels, pos_mask, label_names)

        # 8. Build result payload
        result = {
            "lamhat": float(lamhat),
            "alpha": float(job.alpha),
            "n_samples": len(img_paths),
            "labels": label_names,
            "preprocessing": preproc if preproc is not None else {"note": "no config uploaded — images used at native size"},
            "metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
            },
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
        }

        result_path = result_dir / "lamhat.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        job.result_json = json.dumps(result)
        job.status = JobStatus.DONE
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        db.query(CalibrationJob).filter(CalibrationJob.id == job_id).update(
            {
                "status": JobStatus.FAILED,
                "error_message": str(exc),
                "completed_at": datetime.now(timezone.utc),
            }
        )
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 6. TTL cleanup
# ---------------------------------------------------------------------------

def cleanup_expired_jobs() -> None:
    """Delete jobs and their files that have passed their TTL."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        expired = (
            db.query(CalibrationJob)
            .filter(CalibrationJob.expires_at < now)
            .all()
        )
        for job in expired:
            d = job_dir(job.id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            db.delete(job)
        if expired:
            db.commit()
    finally:
        db.close()
