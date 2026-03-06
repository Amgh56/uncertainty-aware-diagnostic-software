"""Developer route handlers: register, calibration job CRUD."""

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from auth import require_developer
from database import get_db
from enums import UserRole
from models import Doctor
from schemas import (
    DeveloperRegisterRequest,
    ErrorResponse,
    JobCreateResponse,
    JobListResponse,
    JobStatusResponse,
    TokenResponse,
)
from services.auth_service import register_doctor
from services.calibration_service import (
    create_job as svc_create_job,
    delete_job as svc_delete_job,
    get_job as svc_get_job,
    get_job_result as svc_get_job_result,
    list_jobs as svc_list_jobs,
    run_calibration_job,
)

router = APIRouter(tags=["Developer"])


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
    return register_doctor(body.email, body.password, body.full_name, db, role=UserRole.DEVELOPER)


@router.post(
    "/developer/jobs",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload model + config + dataset and start calibration",
    description=(
        "Upload a `.pt` / `.pth` model, an optional `config.json` describing "
        "the model's expected input, and a `.zip` calibration dataset.\n\n"
        "**config.json fields:**\n"
        "```json\n"
        '{ "width": 224, "height": 224, "pixel_mean": 128.0, "pixel_std": 64.0 }\n'
        "```\n"
        "Optional: `use_equalizeHist` (bool, default false).\n\n"
        "**Dataset zip:**\n"
        "```\n"
        "dataset.zip\n"
        "├── images/      (PNG / JPEG files)\n"
        "└── labels.csv   (columns: filename + any label columns)\n"
        "```\n"
        "Minimum 50 labelled images.\n\n"
        "**Size limits:** model ≤ 500 MB, dataset ≤ 2 GB."
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
    job = await svc_create_job(model_file, dataset_file, config_file, alpha, developer, db)
    background_tasks.add_task(run_calibration_job, job.id)
    return job


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
    return JobListResponse(jobs=svc_list_jobs(developer, db))


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
    return svc_get_job(job_id, developer, db)


@router.get(
    "/developer/jobs/{job_id}/result",
    summary="Download lamhat.json",
    description=(
        "Download the `lamhat.json` result file for a completed calibration job. "
        "Contains the calibrated threshold, alpha, number of samples, "
        "and evaluation metrics."
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
    return svc_get_job_result(job_id, developer, db)


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
    svc_delete_job(job_id, developer, db)
