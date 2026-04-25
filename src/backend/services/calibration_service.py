import io
import json
import uuid
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from database import SessionLocal
from enums import JobStatus
from models import CalibrationJob, User
from azure_client import (
    BUCKET_CALIBRATION,
    BUCKET_MODELS,
    delete_from_bucket,
    download_from_bucket,
    upload_to_bucket,
)

MIN_IMAGES = 50
MODEL_SIZE_LIMIT = 500 * 1024 * 1024
DATASET_SIZE_LIMIT = 5 * 1024 * 1024 * 1024

ALLOWED_MODEL_EXT = (".pth", ".pt")
REQUIRED_CONFIG_FIELDS = ("width", "height", "pixel_mean", "pixel_std")



def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return "mps"



# ── Main Mathmatical Functions ────────────
def false_negative_rate(pred_set: np.ndarray, gt_labels: np.ndarray, pos_mask: np.ndarray) -> float:
    pred_set = pred_set[pos_mask]
    gt_labels = gt_labels[pos_mask]
    denom = gt_labels.sum(axis=1)
    if (denom == 0).any():
        raise ValueError("pos_mask still contains rows with zero positives. Something is wrong.")
    num = (pred_set * gt_labels).sum(axis=1)
    recall = num / denom
    return float(1.0 - recall.mean())


def compute_lamhat(cal_sgmd, cal_labels, cal_pos_mask, alpha):
    sgmd = cal_sgmd[cal_pos_mask]
    labels = cal_labels[cal_pos_mask]
    n = sgmd.shape[0]
    target = ((n + 1) / n) * alpha - (1 / n)
    candidates = np.unique(sgmd)
    candidates.sort()
    best = candidates[0]
    for lam in candidates:
        pred_set = sgmd >= lam
        if false_negative_rate(pred_set, labels, np.ones(n, dtype=bool)) <= target:
            best = lam
    return float(best)


# ── Storage path helpers ────

def job_path(job_id: str, filename: str) -> str:
    return f"{job_id}/{filename}"


def result_path(job_id: str, filename: str) -> str:
    return f"{job_id}/result/{filename}"


# ── Helpers ────────────

def get_own_job(job_id: str, developer_id: int, db: Session) -> CalibrationJob:
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


def validate_upload_format(model_file, dataset_file, config_file):
    model_name = (model_file.filename or "").lower()
    if not model_name.endswith(ALLOWED_MODEL_EXT):
        raise HTTPException(status_code=400, detail="Model file must be a .pth or .pt file")
    if not (dataset_file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Dataset file must be a .zip file")
    if config_file is not None and not (config_file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Config file must be a .json file")


# ── Job CRUD ────────

async def create_job(
    model_file,
    dataset_file,
    config_file,
    alpha: float,
    developer: User,
    db: Session,
) -> CalibrationJob:
    """Validate uploads, persist files to Azure, create DB record."""

    validate_upload_format(model_file, dataset_file, config_file)
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
        raise HTTPException(status_code=400, detail="Dataset file exceeds 5 GB limit")

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

    # Generate a human-readable job id 
    job_count = db.query(CalibrationJob).filter(
        CalibrationJob.developer_id == developer.id
    ).count()
    display_name = f"Calibration_Job{job_count + 1}"

    # Upload files to azure blob storage
    upload_to_bucket(BUCKET_CALIBRATION, job_path(job_id, "model.pth"), model_bytes)
    upload_to_bucket(BUCKET_CALIBRATION, job_path(job_id, "dataset.zip"), dataset_bytes, "application/zip")
    if config_bytes is not None:
        upload_to_bucket(BUCKET_CALIBRATION, job_path(job_id, "config.json"), config_bytes, "application/json")

    now = datetime.now(timezone.utc)
    job = CalibrationJob(
        id=job_id,
        display_name=display_name,
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


def list_jobs(developer: User, db: Session) -> list[CalibrationJob]:
    """Return all calibration jobs for a developer, newest first."""
    return (
        db.query(CalibrationJob)
        .filter(CalibrationJob.developer_id == developer.id)
        .order_by(CalibrationJob.created_at.desc())
        .all()
    )


def get_job(job_id: str, developer: User, db: Session) -> CalibrationJob:
    """Return a single job, scoped to the developer."""
    return get_own_job(job_id, developer.id, db)


def get_job_result(job_id: str, developer: User, db: Session) -> Response:
    """Download lamhat.json from Azure for a completed job."""
    job = get_own_job(job_id, developer.id, db)

    if job.status != JobStatus.DONE.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status: {job.status})",
        )

    try:
        data = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, "lamhat.json"))
    except Exception:
        raise HTTPException(status_code=404, detail="Result file not found")

    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="lamhat_{job.display_name or job_id[:8]}.json"'},
    )


def delete_job(job_id: str, developer: User, db: Session) -> None:
    """Delete job record, published model (if any), and files from Azure."""
    job = get_own_job(job_id, developer.id, db)

    # Clean up published model files from Azure if this job was published
    if job.published_model:
        model_id = job.published_model.id
        try:
            delete_from_bucket(BUCKET_MODELS, [
                f"{model_id}/model.pth",
                f"{model_id}/config.json",
            ])
        except Exception:
            pass

    # Clean up calibration job files from Azure
    paths_to_delete = [
        job_path(job_id, "model.pth"),
        job_path(job_id, "dataset.zip"),
        job_path(job_id, "config.json"),
        result_path(job_id, "lamhat.json"),
        result_path(job_id, "cal_probs.npy"),
        result_path(job_id, "cal_labels.npy"),
        result_path(job_id, "cal_pos_mask.npy"),
        result_path(job_id, "label_names.json"),
    ]
    try:
        delete_from_bucket(BUCKET_CALIBRATION, paths_to_delete)
    except Exception:
        pass

    db.delete(job)
    db.commit()


# ── Processing helpers  ────────

def parse_config_bytes(config_bytes: bytes | None) -> dict | None:
    """Parse preprocessing config from raw bytes."""
    if config_bytes is None:
        return None
    cfg = json.loads(config_bytes)
    return {
        "width":            int(cfg["width"]),
        "height":           int(cfg["height"]),
        "pixel_mean":       float(cfg["pixel_mean"]),
        "pixel_std":        float(cfg["pixel_std"]),
        "use_equalizeHist": bool(cfg.get("use_equalizeHist", False)),
    }


def load_model_from_bytes(model_bytes: bytes) -> torch.nn.Module:
    """Load a PyTorch model from raw bytes via BytesIO."""
    buffer = io.BytesIO(model_bytes)

    # Try TorchScript first
    try:
        buffer.seek(0)
        model = torch.jit.load(buffer, map_location="cpu")
        model.eval()
        return model
    except Exception:
        pass

    # Try full saved model
    try:
        buffer.seek(0)
        obj = torch.load(buffer, map_location="cpu", weights_only=False)
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


def preprocess_image_bytes(img_bytes: bytes, preproc: dict | None) -> np.ndarray:
    """Decode raw image bytes and return a float32 array of shape (3, H, W) for PyTorch.

    The function automatically distinguishes grayscale and colour images from the
    decoded data. Grayscale images are resized, optionally equalised, normalised
    using pixel_mean/pixel_std, and stacked to 3 channels. Colour images are
    converted from BGR to RGB, resized, normalised with ImageNet statistics, and
    reordered to channel-first format.

    If no preprocessing config is provided, default size 224x224 is used.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")

    w = int(preproc["width"])  if preproc else 224
    h = int(preproc["height"]) if preproc else 224

    is_grayscale = np.array_equal(bgr[:, :, 0], bgr[:, :, 1]) and \
                   np.array_equal(bgr[:, :, 1], bgr[:, :, 2])

    if is_grayscale:
        gray = bgr[:, :, 0].astype(np.float32)
        gray = cv2.resize(gray, (w, h))
        if preproc and preproc.get("use_equalizeHist", False):
            gray = cv2.equalizeHist(gray.astype(np.uint8)).astype(np.float32)
        mean = float(preproc["pixel_mean"]) if preproc else 128.0
        std  = float(preproc["pixel_std"])  if preproc else 64.0
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


def infer_all_probs_from_bytes(
    model: torch.nn.Module,
    preproc: dict | None,
    image_bytes_list: list[bytes],
    batch_size: int = 16,
) -> np.ndarray:
    """Batched inference over image bytes. Returns probabilities (N, n_classes)."""
    device = pick_device()
    model = model.to(device)

    all_probs = []
    for i in range(0, len(image_bytes_list), batch_size):
        batch = [preprocess_image_bytes(b, preproc) for b in image_bytes_list[i:i + batch_size]]
        x = torch.tensor(np.stack(batch), dtype=torch.float32, device=device)
        all_probs.append(forward(model, x))

    return np.concatenate(all_probs, axis=0)


def evaluate_sets(
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


def extract_dataset_from_zip_bytes(zip_bytes: bytes) -> tuple[list[bytes], np.ndarray, np.ndarray, list[str]]:
    """
    Extract and validate a dataset ZIP from bytes in memory.
    Returns (image_bytes_list, labels, pos_mask, label_names).
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = zf.namelist()

        for member in members:
            if member.startswith("/") or ".." in member:
                raise ValueError(f"Unsafe path in zip: {member!r}")

        # Detect wrapper directory
        top_dirs = {m.split("/")[0] for m in members if "/" in m}
        has_wrapper = (
            len(top_dirs) == 1
            and all(m.startswith(list(top_dirs)[0] + "/") for m in members if m != list(top_dirs)[0] + "/")
        )
        prefix = list(top_dirs)[0] + "/" if has_wrapper else ""

        # Find and read labels.csv
        labels_csv_path = f"{prefix}labels.csv"
        if labels_csv_path not in members:
            raise ValueError("Dataset zip must contain a 'labels.csv' file.")

        labels_csv_bytes = zf.read(labels_csv_path)
        df = pd.read_csv(io.BytesIO(labels_csv_bytes))

        if "filename" not in df.columns:
            raise ValueError("labels.csv must have a 'filename' column.")

        label_names = [c for c in df.columns if c != "filename"]
        if not label_names:
            raise ValueError("labels.csv must have at least one label column besides 'filename'.")

        if len(df) < MIN_IMAGES:
            raise ValueError(
                f"Dataset must have at least {MIN_IMAGES} labelled images (got {len(df)})."
            )

        # Read each image referenced in labels.csv
        image_bytes_list = []
        for fname in df["filename"]:
            img_zip_path = f"{prefix}images/{fname}"
            if img_zip_path not in members:
                raise ValueError(f"Image referenced in labels.csv not found in zip: {fname}")
            image_bytes_list.append(zf.read(img_zip_path))

    labels = df[label_names].values.astype(np.float32)
    pos_mask = (labels.sum(axis=1) > 0)

    return image_bytes_list, labels, pos_mask, label_names


# ── Main calibration pipeline (job worker) ─────

def run_calibration_job(job_id: str) -> None:
    """
    Run the full calibration pipeline for a job.
    Called as a FastAPI BackgroundTask — updates DB status directly.
    """
    # Phase 1: read job metadata and mark as running — close session immediately after
    db = SessionLocal()
    try:
        job = db.query(CalibrationJob).filter(CalibrationJob.id == job_id).first()
        if job is None:
            return
        alpha = job.alpha
        config_filename = job.config_filename
        job.status = JobStatus.RUNNING.value
        db.commit()
    except Exception as exc:
        try:
            db.query(CalibrationJob).filter(CalibrationJob.id == job_id).update({
                "status": JobStatus.FAILED.value,
                "error_message": str(exc),
                "completed_at": datetime.now(timezone.utc),
            })
            db.commit()
        except Exception:
            pass
        return
    finally:
        db.close()

    # Phase 2: all heavy work runs with no DB session open
    try:
        model_bytes = download_from_bucket(BUCKET_CALIBRATION, job_path(job_id, "model.pth"))
        dataset_bytes = download_from_bucket(BUCKET_CALIBRATION, job_path(job_id, "dataset.zip"))

        config_bytes = None
        if config_filename:
            config_bytes = download_from_bucket(BUCKET_CALIBRATION, job_path(job_id, "config.json"))

        preproc = parse_config_bytes(config_bytes)
        image_bytes_list, labels, pos_mask, label_names = extract_dataset_from_zip_bytes(dataset_bytes)

        model = load_model_from_bytes(model_bytes)
        probs = infer_all_probs_from_bytes(model, preproc, image_bytes_list, batch_size=16)
        lamhat = compute_lamhat(probs, labels, pos_mask, alpha)

        pred_sets = (probs >= lamhat).astype(np.float32)
        metrics = evaluate_sets(pred_sets, labels, pos_mask, label_names)

        result = {
            "lamhat": float(lamhat),
            "alpha": float(alpha),
            "n_samples": len(image_bytes_list),
            "labels": label_names,
            "preprocessing": preproc if preproc is not None else {"note": "no config uploaded"},
            "metrics": {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
            },
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
        }

        upload_npy(job_id, "cal_probs.npy", probs)
        upload_npy(job_id, "cal_labels.npy", labels)
        upload_npy(job_id, "cal_pos_mask.npy", pos_mask)
        upload_to_bucket(
            BUCKET_CALIBRATION,
            result_path(job_id, "label_names.json"),
            json.dumps(label_names).encode(),
            "application/json",
        )
        upload_to_bucket(
            BUCKET_CALIBRATION,
            result_path(job_id, "lamhat.json"),
            json.dumps(result, indent=2).encode(),
            "application/json",
        )

        # Phase 3: write results with a fresh session
        db = SessionLocal()
        try:
            db.query(CalibrationJob).filter(CalibrationJob.id == job_id).update({
                "result_json": json.dumps(result),
                "status": JobStatus.DONE.value,
                "completed_at": datetime.now(timezone.utc),
            })
            db.commit()
        finally:
            db.close()

    except Exception as exc:
        db = SessionLocal()
        try:
            db.query(CalibrationJob).filter(CalibrationJob.id == job_id).update({
                "status": JobStatus.FAILED.value,
                "error_message": str(exc),
                "completed_at": datetime.now(timezone.utc),
            })
            db.commit()
        finally:
            db.close()


def upload_npy(job_id: str, filename: str, arr: np.ndarray) -> None:
    """Serialize a numpy array and upload to Azure."""
    buf = io.BytesIO()
    np.save(buf, arr)
    upload_to_bucket(BUCKET_CALIBRATION, result_path(job_id, filename), buf.getvalue())


# ─── Validation Feature ───────────────────────────────────────────────

VALIDATION_ALPHAS = np.linspace(0.001, 0.99, 200).tolist()


def load_validation_artifacts(job_id: str):
    """Download saved numpy artifacts from Azure for a completed job."""
    try:
        probs_bytes = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, "cal_probs.npy"))
    except Exception:
        return None

    labels_bytes = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, "cal_labels.npy"))
    mask_bytes = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, "cal_pos_mask.npy"))
    names_bytes = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, "label_names.json"))

    probs = np.load(io.BytesIO(probs_bytes))
    labels = np.load(io.BytesIO(labels_bytes))
    pos_mask = np.load(io.BytesIO(mask_bytes))
    label_names = json.loads(names_bytes)

    return probs, labels, pos_mask, label_names


def compute_validation_sweep(probs, labels, pos_mask, label_names, job_alpha):
    """Sweep across alphas and compute FNR + avg set size at each."""
    sweep = []
    for a in VALIDATION_ALPHAS:
        lam = compute_lamhat(probs, labels, pos_mask, float(a))
        pred_sets = (probs >= lam).astype(np.float32)
        fnr = false_negative_rate(pred_sets, labels, pos_mask)
        avg_size = float(pred_sets.sum(axis=1).mean())
        sweep.append({
            "alpha": round(float(a), 4),
            "lamhat": round(float(lam), 6),
            "empirical_fnr": round(float(fnr), 4),
            "avg_set_size": round(avg_size, 4),
        })

    # Compute metrics at the job's own alpha
    job_lam = compute_lamhat(probs, labels, pos_mask, float(job_alpha))
    job_preds = (probs >= job_lam).astype(np.float32)
    job_fnr = false_negative_rate(job_preds, labels, pos_mask)
    job_avg_size = float(job_preds.sum(axis=1).mean())

    # Quality verdict
    fnr_values = [s["empirical_fnr"] for s in sweep]
    alpha_values = [s["alpha"] for s in sweep]
    violations = sum(1 for f, a in zip(fnr_values, alpha_values) if f > a + 0.05)
    avg_sizes = [s["avg_set_size"] for s in sweep]
    monotonic_breaks = sum(
        1 for i in range(1, len(avg_sizes)) if avg_sizes[i] > avg_sizes[i - 1] + 0.1
    )

    if violations <= 2 and monotonic_breaks <= 3 and job_fnr <= job_alpha + 0.02:
        verdict = "good"
    elif violations <= 8 and job_fnr <= job_alpha + 0.1:
        verdict = "review"
    else:
        verdict = "unreliable"

    return {
        "sweep": sweep,
        "job_alpha": round(float(job_alpha), 4),
        "job_lamhat": round(float(job_lam), 6),
        "job_fnr": round(float(job_fnr), 4),
        "job_avg_set_size": round(float(job_avg_size), 4),
        "n_samples": int(probs.shape[0]),
        "n_positive": int(pos_mask.sum()),
        "label_names": label_names,
        "verdict": verdict,
        "violations": violations,
        "monotonic_breaks": monotonic_breaks,
    }


_VALIDATION_CACHE = "validation_result.json"


def get_validation_data(job_id: str, developer: User, db: Session) -> dict:
    """Return validation sweep data for a completed job.

    The sweep result is cached as validation_result.json after the first call
    so all subsequent requests return instantly without re-running the sweep.
    """
    job = get_own_job(job_id, developer.id, db)

    if job.status != JobStatus.DONE.value:
        raise HTTPException(status_code=400, detail=f"Job is not complete (status: {job.status})")

    # Fast path — blob exists if verdict is set (true for all jobs after upload-first fix)
    if job.validation_verdict:
        try:
            cached_bytes = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, _VALIDATION_CACHE))
            return json.loads(cached_bytes)
        except Exception:
            # Blob missing for old jobs where verdict was written before blob was uploaded.
            # Clear the stale verdict and fall through to recompute below.
            job.validation_verdict = None
            db.commit()

    artifacts = load_validation_artifacts(job_id)
    if artifacts is None:
        raise HTTPException(
            status_code=404,
            detail="Validation artifacts not found. Please regenerate.",
        )

    probs, labels, pos_mask, label_names = artifacts
    result = compute_validation_sweep(probs, labels, pos_mask, label_names, job.alpha)

    # Upload blob first — verdict is only written after blob is confirmed uploaded
    upload_to_bucket(
        BUCKET_CALIBRATION,
        result_path(job_id, _VALIDATION_CACHE),
        json.dumps(result).encode(),
        "application/json",
    )

    job.validation_verdict = result["verdict"]
    db.commit()

    return result


def get_validation_artifact(job_id: str, filename: str, developer: User, db: Session) -> tuple[bytes, str]:
    """Download a validation artifact from Azure. Returns (bytes, media_type)."""
    job = get_own_job(job_id, developer.id, db)

    if job.status != JobStatus.DONE.value:
        raise HTTPException(status_code=400, detail="Job is not complete")

    allowed = {"cal_probs.npy", "cal_labels.npy", "cal_pos_mask.npy", "label_names.json"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid artifact: {filename}")

    try:
        data = download_from_bucket(BUCKET_CALIBRATION, result_path(job_id, filename))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")

    media_type = "application/json" if filename.endswith(".json") else "application/octet-stream"
    return data, media_type
