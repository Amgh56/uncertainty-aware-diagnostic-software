"""
Developer / Researcher route handlers.

Endpoints:
  POST   /developer/register           — Create a developer account
  POST   /developer/jobs               — Upload model + dataset, start calibration
  GET    /developer/jobs               — List own jobs
  GET    /developer/jobs/{job_id}      — Get single job status
  GET    /developer/jobs/{job_id}/result — Download lamhat.json
  DELETE /developer/jobs/{job_id}      — Delete job + files
"""

import shutil
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth import create_access_token, get_current_doctor, hash_password, require_developer
from database import get_db
from models import CalibrationJob, Doctor, JobStatus, UserRole
from schemas import (
    DeveloperRegisterRequest,
    ErrorResponse,
    JobCreateResponse,
    JobListResponse,
    JobStatusResponse,
    TokenResponse,
)
from services.calibration_service import (
    MODEL_SIZE_LIMIT,
    DATASET_SIZE_LIMIT,
    TTL_DAYS,
    cleanup_expired_jobs,
    job_dir,
    run_calibration_job,
)

router = APIRouter(tags=["Developer"])

# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


@router.post(
    "/developer/register",
    response_model=TokenResponse,
    summary="Register a developer / researcher account",
    description=(
        "Create a new account with the `developer` role. "
        "Developers can upload pretrained models and calibration datasets to run "
        "the conformal calibration pipeline."
    ),
    responses={
        200: {"description": "Registration successful, JWT token returned"},
        400: {
            "model": ErrorResponse,
            "description": "Email already registered or password too short",
        },
    },
)
def register_developer(body: DeveloperRegisterRequest, db: Session = Depends(get_db)):
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = db.query(Doctor).filter(Doctor.email == body.email.lower().strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    doctor = Doctor(
        email=body.email.lower().strip(),
        hashed_password=hash_password(body.password),
        full_name=body.full_name.strip(),
        role=UserRole.DEVELOPER,
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    token = create_access_token(doctor.id)
    return TokenResponse(access_token=token)


# ---------------------------------------------------------------------------
# Create job
# ---------------------------------------------------------------------------


@router.post(
    "/developer/jobs",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload model + config + dataset and start calibration",
    description=(
        "Upload a `.pth` model, a `config.json` describing the model's expected input, "
        "and a `.zip` calibration dataset. The pipeline rescales images to match the "
        "model's expected dimensions from the config.\n\n"
        "**config.json required fields:**\n"
        "```json\n"
        '{ "width": 224, "height": 224, "pixel_mean": 128.0, "pixel_std": 64.0 }\n'
        "```\n"
        "Optional: `use_equalizeHist` (bool, default false).\n\n"
        "**Dataset zip spec:**\n"
        "```\n"
        "dataset.zip\n"
        "├── images/      (PNG / JPEG files)\n"
        "└── labels.csv   (columns: filename + any label columns)\n"
        "```\n"
        "Minimum 50 labelled images.\n\n"
        f"**Size limits:** model ≤ 500 MB, dataset ≤ 2 GB."
    ),
    responses={
        201: {"description": "Job created and queued"},
        400: {"model": ErrorResponse, "description": "File too large or wrong extension"},
        401: {"model": ErrorResponse, "description": "Invalid or expired token"},
        403: {"model": ErrorResponse, "description": "Developer access required"},
    },
)
async def create_job(
    background_tasks: BackgroundTasks,
    model_file: UploadFile,
    dataset_file: UploadFile,
    config_file: UploadFile = None,
    alpha: float = 0.1,
    db: Session = Depends(get_db),
    developer: Doctor = Depends(require_developer),
):
    # --- validate extensions ---
    model_name = (model_file.filename or "").lower()
    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        raise HTTPException(status_code=400, detail="Model file must be a .pth or .pt file")
    if not (dataset_file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Dataset file must be a .zip file")
    if config_file is not None and not (config_file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Config file must be a .json file")

    if not (0.0 < alpha < 1.0):
        raise HTTPException(status_code=400, detail="alpha must be between 0 and 1 (exclusive)")

    # --- run TTL cleanup before accepting a new job ---
    cleanup_expired_jobs()

    # --- read files into memory + enforce size limits ---
    model_bytes = await model_file.read()
    if len(model_bytes) > MODEL_SIZE_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"Model file exceeds 500 MB limit ({len(model_bytes) // (1024*1024)} MB uploaded)",
        )

    dataset_bytes = await dataset_file.read()
    if len(dataset_bytes) > DATASET_SIZE_LIMIT:
        raise HTTPException(
            status_code=400,
            detail="Dataset file exceeds 2 GB limit",
        )

    # --- validate config JSON if provided ---
    import json as _json
    config_bytes = None
    config_fname = None
    if config_file is not None:
        config_bytes = await config_file.read()
        config_fname = config_file.filename
        try:
            cfg = _json.loads(config_bytes)
        except Exception:
            raise HTTPException(status_code=400, detail="config_file is not valid JSON")
        # Required preprocessing fields when a config is uploaded
        for field in ("width", "height", "pixel_mean", "pixel_std"):
            if field not in cfg:
                raise HTTPException(
                    status_code=400,
                    detail=f"config.json is missing required field: '{field}'"
                )

    # --- persist to disk ---
    job_id = str(uuid.uuid4())
    base = job_dir(job_id)
    base.mkdir(parents=True, exist_ok=True)

    (base / "model.pth").write_bytes(model_bytes)
    (base / "dataset.zip").write_bytes(dataset_bytes)
    if config_bytes is not None:
        (base / "config.json").write_bytes(config_bytes)

    # --- create DB record ---
    now = datetime.now(timezone.utc)
    job = CalibrationJob(
        id=job_id,
        developer_id=developer.id,
        status=JobStatus.QUEUED,
        model_filename=model_file.filename,
        config_filename=config_fname,
        dataset_filename=dataset_file.filename,
        alpha=alpha,
        created_at=now,
        expires_at=now + timedelta(days=TTL_DAYS),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # --- dispatch background task ---
    background_tasks.add_task(run_calibration_job, job_id)

    return job


# ---------------------------------------------------------------------------
# List jobs
# ---------------------------------------------------------------------------


@router.get(
    "/developer/jobs",
    response_model=JobListResponse,
    summary="List calibration jobs",
    description="Return all calibration jobs belonging to the authenticated developer, newest first.",
    responses={
        200: {"description": "List of jobs returned"},
        401: {"model": ErrorResponse, "description": "Invalid or expired token"},
        403: {"model": ErrorResponse, "description": "Developer access required"},
    },
)
def list_jobs(
    db: Session = Depends(get_db),
    developer: Doctor = Depends(require_developer),
):
    jobs = (
        db.query(CalibrationJob)
        .filter(CalibrationJob.developer_id == developer.id)
        .order_by(CalibrationJob.created_at.desc())
        .all()
    )
    return JobListResponse(jobs=jobs)


# ---------------------------------------------------------------------------
# Get single job status
# ---------------------------------------------------------------------------


@router.get(
    "/developer/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get calibration job status",
    description="Return the full status and metadata for a single calibration job.",
    responses={
        200: {"description": "Job status returned"},
        401: {"model": ErrorResponse, "description": "Invalid or expired token"},
        403: {"model": ErrorResponse, "description": "Developer access required"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    developer: Doctor = Depends(require_developer),
):
    job = _get_own_job(job_id, developer.id, db)
    return job


# ---------------------------------------------------------------------------
# Download result
# ---------------------------------------------------------------------------


@router.get(
    "/developer/jobs/{job_id}/result",
    summary="Download lamhat.json",
    description=(
        "Download the `lamhat.json` result file for a completed calibration job. "
        "Contains the calibrated threshold (λ̂), alpha, number of samples, "
        "and evaluation metrics (marginal coverage, average set size, empty-set rate)."
    ),
    responses={
        200: {"description": "lamhat.json file returned"},
        400: {"model": ErrorResponse, "description": "Job is not yet completed"},
        401: {"model": ErrorResponse, "description": "Invalid or expired token"},
        403: {"model": ErrorResponse, "description": "Developer access required"},
        404: {"model": ErrorResponse, "description": "Job or result file not found"},
    },
)
def download_result(
    job_id: str,
    db: Session = Depends(get_db),
    developer: Doctor = Depends(require_developer),
):
    job = _get_own_job(job_id, developer.id, db)

    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status: {job.status})",
        )

    result_path = job_dir(job_id) / "result" / "lamhat.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    return FileResponse(
        path=str(result_path),
        media_type="application/json",
        filename=f"lamhat_{job_id[:8]}.json",
    )


# ---------------------------------------------------------------------------
# Delete job
# ---------------------------------------------------------------------------


@router.delete(
    "/developer/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a calibration job",
    description="Delete the calibration job record and all uploaded files from disk.",
    responses={
        204: {"description": "Job deleted successfully"},
        401: {"model": ErrorResponse, "description": "Invalid or expired token"},
        403: {"model": ErrorResponse, "description": "Developer access required"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    developer: Doctor = Depends(require_developer),
):
    job = _get_own_job(job_id, developer.id, db)

    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

    db.delete(job)
    db.commit()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _get_own_job(job_id: str, developer_id: int, db: Session) -> CalibrationJob:
    """Fetch a job by ID, ensuring it belongs to the given developer."""
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
