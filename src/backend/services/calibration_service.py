import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from conformal_prediction_pipeline import (
    compute_lamhat,
    false_negative_rate,
    pick_device,
)
from database import SessionLocal
from enums import JobStatus
from models import CalibrationJob, Doctor
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BACKEND_DIR / "developer_uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

MIN_IMAGES = 50
MODEL_SIZE_LIMIT = 500 * 1024 * 1024       
DATASET_SIZE_LIMIT = 2 * 1024 * 1024 * 1024  

ALLOWED_MODEL_EXT = (".pth", ".pt")
REQUIRED_CONFIG_FIELDS = ("width", "height", "pixel_mean", "pixel_std")




def _job_dir(job_id: str) -> Path:
    return UPLOADS_DIR / job_id


def _get_own_job(job_id: str, developer_id: int, db: Session) -> CalibrationJob:
    job = (
        db.query(CalibrationJob)
        .filter(
            CalibrationJob.id == job_id,
            CalibrationJob.developer_id == developer_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _validate_upload_format(model_file, dataset_file, config_file):
    model_name = (model_file.filename or "").lower()
    if not model_name.endswith(ALLOWED_MODEL_EXT):
        raise HTTPException(status_code=400, detail="Model file must be a .pth or .pt file")
    if not (dataset_file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Dataset file must be a .zip file")
    if config_file is not None and not (config_file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Config file must be a .json file")
    

async def create_job(
    model_file,
    dataset_file,
    config_file,
    alpha: float,
    developer: Doctor,
    db: Session,
) -> CalibrationJob:
    """Validate uploads, persist files to disk, create DB record."""
    
    _validate_upload_format(model_file, dataset_file, config_file)
    if not (0.0 < alpha < 1.0):
        raise HTTPException(status_code=400, detail="alpha must be between 0 and 1")

    model_bytes = await model_file.read()
    if len(model_bytes) > MODEL_SIZE_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"Model file exceeds 500 MB limit ({len(model_bytes) // (1024*1024)} MB uploaded)",
        )

    dataset_bytes = await dataset_file.read()
    if len(dataset_bytes) > DATASET_SIZE_LIMIT:
        raise HTTPException(status_code=400, detail="Dataset file exceeds 2 GB limit")

    config_bytes = None
    config_fname = None
    if config_file is not None:
        config_bytes = await config_file.read()
        config_fname = config_file.filename
        try:
            cfg = json.loads(config_bytes)
        except Exception:
            raise HTTPException(status_code=400, detail="config_file is not valid JSON")
        for field in REQUIRED_CONFIG_FIELDS:
            if field not in cfg:
                raise HTTPException(
                    status_code=400,
                    detail=f"config.json is missing required field: '{field}'",
                )

    job_id = str(uuid.uuid4())
    base = _job_dir(job_id)
    base.mkdir(parents=True, exist_ok=True)

    (base / "model.pth").write_bytes(model_bytes)
    (base / "dataset.zip").write_bytes(dataset_bytes)
    if config_bytes is not None:
        (base / "config.json").write_bytes(config_bytes)

    now = datetime.now(timezone.utc)
    job = CalibrationJob(
        id=job_id,
        developer_id=developer.id,
        status=JobStatus.QUEUED.value,
        model_filename=model_file.filename,
        config_filename=config_fname,
        dataset_filename=dataset_file.filename,
        alpha=alpha,
        created_at=now,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def list_jobs(developer: Doctor, db: Session) -> list[CalibrationJob]:
    """Return all calibration jobs for a developer, newest first."""
    return (
        db.query(CalibrationJob)
        .filter(CalibrationJob.developer_id == developer.id)
        .order_by(CalibrationJob.created_at.desc())
        .all()
    )


def get_job(job_id: str, developer: Doctor, db: Session) -> CalibrationJob:
    """Return a single job, scoped to the developer."""
    return _get_own_job(job_id, developer.id, db)


def get_job_result(job_id: str, developer: Doctor, db: Session) -> FileResponse:
    """Return a FileResponse for the lamhat.json of a completed job."""
    job = _get_own_job(job_id, developer.id, db)

    if job.status != JobStatus.DONE.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status: {job.status})",
        )

    result_path = _job_dir(job_id) / "result" / "lamhat.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename=f"lamhat_{job_id[:8]}.json",
    )


def delete_job(job_id: str, developer: Doctor, db: Session) -> None:
    """Delete job record + files from disk."""
    job = _get_own_job(job_id, developer.id, db)

    d = _job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

    db.delete(job)
    db.commit()




def _read_preproc(config_path: Path) -> dict | None:
    """
    Read preprocessing parameters from the uploaded config.json.

    Required fields (when config is present): width, height, pixel_mean, pixel_std.
    Optional: use_equalizeHist (default False).

    Returns None if no config was uploaded.
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




def _load_generic_model(model_path: Path) -> torch.nn.Module:
    """
    Load a model file.  Two formats are accepted:

    1. TorchScript (recommended) — torch.jit.save / torch.jit.load
    2. Full nn.Module — torch.save(model, path)

    State dicts (OrderedDict) are not accepted.
    """
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        pass

    try:
        obj = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not read model file: {exc}") from exc

    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

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




def _preprocess_image(img_path: Path, preproc: dict | None) -> np.ndarray:
    """Read a grayscale image and return a float32 array (3, H, W)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = img.astype(np.float32)

    if preproc is not None:
        img = cv2.resize(img, (preproc["width"], preproc["height"]))
        if preproc.get("use_equalizeHist", False):
            img = cv2.equalizeHist(img.astype(np.uint8)).astype(np.float32)
        img = (img - preproc["pixel_mean"]) / preproc["pixel_std"]

    return np.stack([img, img, img], axis=0)


def _forward(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Run inference and return sigmoid probabilities as numpy (B, n_classes)."""
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


def _infer_all_probs(
    model: torch.nn.Module,
    preproc: dict | None,
    img_paths: list,
    batch_size: int = 16,
) -> np.ndarray:
    """Batched inference over all images. Returns probabilities (N, n_classes)."""
    device = pick_device()
    model = model.to(device)

    all_probs = []
    for i in range(0, len(img_paths), batch_size):
        batch = [_preprocess_image(p, preproc) for p in img_paths[i:i + batch_size]]
        x = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
        all_probs.append(_forward(model, x))

    return np.concatenate(all_probs, axis=0)




def _evaluate_sets(
    pred_sets: np.ndarray,
    labels: np.ndarray,
    pos_mask: np.ndarray,
    label_names: list,
) -> dict:
    """Compute calibration metrics for any number of labels."""
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



def _validate_and_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Validate zip structure, protect against path traversal, extract."""
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


def _validate_labels_csv(labels_csv: Path, images_dir: Path) -> tuple:
    """Parse labels.csv. Any column other than 'filename' is a label."""

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

        job.status = JobStatus.RUNNING.value
        db.commit()

        base        = _job_dir(job_id)
        model_path  = base / "model.pth"
        config_path = base / "config.json"
        dataset_zip = base / "dataset.zip"
        dataset_dir = base / "dataset"
        result_dir  = base / "result"
        result_dir.mkdir(exist_ok=True)
        dataset_dir.mkdir(exist_ok=True)

        preproc      = _read_preproc(config_path)
        dataset_root = _validate_and_extract_zip(dataset_zip, dataset_dir)

        images_dir = dataset_root / "images"
        labels_csv = dataset_root / "labels.csv"
        img_paths, labels, pos_mask, label_names = _validate_labels_csv(labels_csv, images_dir)

        model = _load_generic_model(model_path)
        model = model.to(pick_device())

        probs   = _infer_all_probs(model, preproc, img_paths, batch_size=16)
        lamhat  = compute_lamhat(probs, labels, pos_mask, job.alpha)

        pred_sets = (probs >= lamhat).astype(np.float32)
        metrics   = _evaluate_sets(pred_sets, labels, pos_mask, label_names)

        result = {
            "lamhat": float(lamhat),
            "alpha": float(job.alpha),
            "n_samples": len(img_paths),
            "labels": label_names,
            "preprocessing": preproc if preproc is not None else {"note": "no config uploaded"},
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
        job.status = JobStatus.DONE.value
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        db.query(CalibrationJob).filter(CalibrationJob.id == job_id).update(
            {
                "status": JobStatus.FAILED.value,
                "error_message": str(exc),
                "completed_at": datetime.now(timezone.utc),
            }
        )
        db.commit()
    finally:
        db.close()




