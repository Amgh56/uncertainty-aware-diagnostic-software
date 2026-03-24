"""Service layer for publishing, listing, and managing Published Model Packages."""

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from enums import JobStatus, ModelVisibility
from models import CalibrationJob, User, PublishedModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
PUBLISHED_DIR = BACKEND_DIR / "published_models"
PUBLISHED_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = BACKEND_DIR / "developer_uploads"

CONSENT_TEXT = (
    "By releasing this model, you acknowledge that: "
    "Your model artifact, metadata, and calibration parameters will be shared "
    "with the selected audience. "
    "For clinician release: this model may be used as a diagnostic aid for patients. "
    "You are responsible for the quality and validity of your model within the scope "
    "of your validation. "
    "Your name will be displayed as the model creator. "
    "You can change visibility or deactivate the model at any time. "
    "Real clinical deployment requires formal review beyond this platform's validation."
)
CONSENT_TEXT_HASH = hashlib.sha256(CONSENT_TEXT.encode()).hexdigest()


def _get_job_dir(job_id: str) -> Path:
    return UPLOADS_DIR / job_id


def publish_model(
    calibration_job_id: str,
    name: str,
    description: str,
    version: str,
    modality: str,
    intended_use: str,
    labels: list[str],
    visibility: ModelVisibility,
    consent_agreed: bool,
    developer: User,
    db: Session,
) -> PublishedModel:
    """Create a Published Model Package from a completed calibration job."""

    # Validate job ownership and state
    job = (
        db.query(CalibrationJob)
        .filter(
            CalibrationJob.id == calibration_job_id,
            CalibrationJob.developer_id == developer.id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Calibration job not found")

    if job.status != JobStatus.DONE.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete (status: {job.status})",
        )

    if job.is_published:
        raise HTTPException(
            status_code=400,
            detail="This calibration job has already been published",
        )

    # Check verdict — block unreliable
    verdict = job.validation_verdict
    if verdict is None:
        raise HTTPException(
            status_code=400,
            detail="Validation has not been run yet. Please validate the calibration first.",
        )
    if verdict == "unreliable":
        raise HTTPException(
            status_code=400,
            detail="Cannot publish a model with 'unreliable' validation verdict. "
            "Please recalibrate with a larger or higher-quality dataset.",
        )

    # Consent required for non-private visibility
    if visibility != ModelVisibility.PRIVATE and not consent_agreed:
        raise HTTPException(
            status_code=400,
            detail="Consent is required for non-private visibility",
        )

    # Extract calibration data from job result
    if not job.result_json:
        raise HTTPException(status_code=400, detail="Job has no result data")

    result_data = json.loads(job.result_json)
    lamhat = result_data["lamhat"]
    alpha = result_data["alpha"]

    # Determine artifact type
    job_dir = _get_job_dir(calibration_job_id)
    model_path = job_dir / "model.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact not found on disk")

    # Detect artifact type (try torchscript first)
    artifact_type = "pytorch"
    try:
        import torch
        torch.jit.load(str(model_path), map_location="cpu")
        artifact_type = "torchscript"
    except Exception:
        pass

    # Copy model to published directory
    model_id = str(uuid.uuid4())
    pub_dir = PUBLISHED_DIR / model_id
    pub_dir.mkdir(parents=True, exist_ok=True)
    dest_model = pub_dir / "model.pth"
    shutil.copy2(str(model_path), str(dest_model))

    # Copy config if exists
    config_path = job_dir / "config.json"
    config_json_str = None
    if config_path.exists():
        shutil.copy2(str(config_path), str(pub_dir / "config.json"))
        with open(config_path) as f:
            config_json_str = f.read()

    # Build validation metrics from result
    metrics = result_data.get("metrics", {})

    now = datetime.now(timezone.utc)
    published = PublishedModel(
        id=model_id,
        calibration_job_id=calibration_job_id,
        developer_id=developer.id,
        name=name,
        description=description,
        version=version,
        modality=modality,
        intended_use=intended_use,
        artifact_path=str(dest_model),
        artifact_type=artifact_type,
        config_json=config_json_str,
        labels_json=json.dumps(labels),
        num_labels=len(labels),
        alpha=alpha,
        lamhat=lamhat,
        lamhat_result_json=job.result_json,
        validation_verdict=verdict,
        validation_metrics_json=json.dumps(metrics),
        visibility=visibility.value,
        is_active=True,
        consent_given_at=now if consent_agreed else None,
        consent_text_hash=CONSENT_TEXT_HASH if consent_agreed else None,
        created_at=now,
        updated_at=now,
    )

    job.is_published = True
    db.add(published)
    db.commit()
    db.refresh(published)
    return published


def list_my_models(developer: User, db: Session) -> list[dict]:
    """List all published models owned by the developer."""
    models = (
        db.query(PublishedModel)
        .filter(PublishedModel.developer_id == developer.id)
        .order_by(PublishedModel.created_at.desc())
        .all()
    )
    return [_model_to_summary(m, developer.full_name) for m in models]


def list_community_models(
    db: Session,
    search: str | None = None,
    modality: str | None = None,
    verdict: str | None = None,
    sort: str = "newest",
) -> list[dict]:
    """List models visible to the developer community."""
    q = db.query(PublishedModel, User.full_name).join(
        User, PublishedModel.developer_id == User.id
    ).filter(
        PublishedModel.visibility.in_([
            ModelVisibility.COMMUNITY.value,
            ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
        ]),
        PublishedModel.is_active == True,
    )

    if search:
        pattern = f"%{search}%"
        q = q.filter(
            (PublishedModel.name.ilike(pattern))
            | (PublishedModel.description.ilike(pattern))
        )
    if modality:
        q = q.filter(PublishedModel.modality == modality)
    if verdict:
        q = q.filter(PublishedModel.validation_verdict == verdict)

    if sort == "alphabetical":
        q = q.order_by(PublishedModel.name.asc())
    else:
        q = q.order_by(PublishedModel.created_at.desc())

    results = q.all()
    return [_model_to_summary(m, dev_name) for m, dev_name in results]


def list_clinician_models(db: Session) -> list[dict]:
    """List models visible to clinicians for inference."""
    results = (
        db.query(PublishedModel, User.full_name)
        .join(User, PublishedModel.developer_id == User.id)
        .filter(
            PublishedModel.visibility.in_([
                ModelVisibility.CLINICIAN.value,
                ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
            ]),
            PublishedModel.is_active == True,
        )
        .order_by(PublishedModel.created_at.desc())
        .all()
    )
    return [_model_to_summary(m, dev_name) for m, dev_name in results]


def get_model_detail(
    model_id: str, requesting_user: User, db: Session
) -> dict:
    """Get full details of a published model with access control."""
    result = (
        db.query(PublishedModel, User.full_name)
        .join(User, PublishedModel.developer_id == User.id)
        .filter(PublishedModel.id == model_id)
        .first()
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Published model not found")

    model, dev_name = result

    # Access control
    is_owner = model.developer_id == requesting_user.id
    if not is_owner:
        vis = model.visibility
        role = requesting_user.role
        if role == "clinician" and vis not in (
            ModelVisibility.CLINICIAN.value,
            ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        if role == "developer" and vis not in (
            ModelVisibility.COMMUNITY.value,
            ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
        ):
            raise HTTPException(status_code=403, detail="Access denied")

    return _model_to_detail(model, dev_name)


def update_visibility(
    model_id: str,
    new_visibility: ModelVisibility,
    consent_agreed: bool,
    developer: User,
    db: Session,
) -> PublishedModel:
    """Change a model's visibility."""
    model = (
        db.query(PublishedModel)
        .filter(
            PublishedModel.id == model_id,
            PublishedModel.developer_id == developer.id,
        )
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Published model not found")

    old_vis = model.visibility
    new_vis = new_visibility.value

    # Expanding visibility requires consent
    expanding = _is_expanding_visibility(old_vis, new_vis)
    if expanding and not consent_agreed:
        raise HTTPException(
            status_code=400,
            detail="Consent is required when expanding model visibility",
        )

    now = datetime.now(timezone.utc)
    model.visibility = new_vis
    if expanding and consent_agreed:
        model.consent_given_at = now
        model.consent_text_hash = CONSENT_TEXT_HASH
    model.updated_at = now
    db.commit()
    db.refresh(model)
    return model


def toggle_active(
    model_id: str,
    is_active: bool,
    developer: User,
    db: Session,
) -> PublishedModel:
    """Activate or deactivate a published model."""
    model = (
        db.query(PublishedModel)
        .filter(
            PublishedModel.id == model_id,
            PublishedModel.developer_id == developer.id,
        )
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Published model not found")

    model.is_active = is_active
    model.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(model)
    return model


def get_model_artifact_path(model_id: str, db: Session) -> Path:
    """Return the path to the model artifact file for download."""
    model = db.query(PublishedModel).filter(PublishedModel.id == model_id).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Published model not found")

    path = Path(model.artifact_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model artifact file not found")

    return path


# ── Helpers ──────────────────────────────────────────────────

def _is_expanding_visibility(old: str, new: str) -> bool:
    """Check if changing from old to new visibility expands the audience."""
    if new == ModelVisibility.PRIVATE.value:
        return False
    if old == ModelVisibility.PRIVATE.value:
        return True
    if old == ModelVisibility.CLINICIAN.value and new in (
        ModelVisibility.COMMUNITY.value,
        ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
    ):
        return True
    if old == ModelVisibility.COMMUNITY.value and new in (
        ModelVisibility.CLINICIAN.value,
        ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
    ):
        return True
    return False


def _model_to_summary(model: PublishedModel, developer_name: str) -> dict:
    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "version": model.version,
        "modality": model.modality,
        "num_labels": model.num_labels,
        "alpha": model.alpha,
        "lamhat": model.lamhat,
        "validation_verdict": model.validation_verdict,
        "visibility": model.visibility,
        "is_active": model.is_active,
        "developer_name": developer_name,
        "created_at": model.created_at.isoformat() if model.created_at else None,
    }


def _model_to_detail(model: PublishedModel, developer_name: str) -> dict:
    return {
        "id": model.id,
        "calibration_job_id": model.calibration_job_id,
        "developer_id": model.developer_id,
        "name": model.name,
        "description": model.description,
        "version": model.version,
        "modality": model.modality,
        "intended_use": model.intended_use,
        "artifact_type": model.artifact_type,
        "labels_json": model.labels_json,
        "num_labels": model.num_labels,
        "alpha": model.alpha,
        "lamhat": model.lamhat,
        "lamhat_result_json": model.lamhat_result_json,
        "validation_verdict": model.validation_verdict,
        "validation_metrics_json": model.validation_metrics_json,
        "visibility": model.visibility,
        "is_active": model.is_active,
        "consent_given_at": model.consent_given_at.isoformat() if model.consent_given_at else None,
        "developer_name": developer_name,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "updated_at": model.updated_at.isoformat() if model.updated_at else None,
    }
