import hashlib
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone

import torch
from fastapi import HTTPException
from sqlalchemy.orm import Session

from enums import ArtifactType, JobStatus, ModelVisibility, UserRole, ValidationVerdict
from models import CalibrationJob, User, PublishedModel
from services.calibration_service import result_path
from azure_client import (
    BUCKET_CALIBRATION,
    BUCKET_MODELS,
    download_from_bucket,
    iter_blob_chunks,
    upload_to_bucket,
)

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
VALIDATION_CACHE_FILENAME = "validation_result.json"


# ── Publishing ───────────────────────────────────────────────

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

    # Check verdict — only good models can be published
    verdict = job.validation_verdict
    if verdict is None:
        raise HTTPException(
            status_code=400,
            detail="Validation has not been run yet. Please validate the calibration first.",
        )
    if verdict != ValidationVerdict.GOOD.value:
        raise HTTPException(
            status_code=400,
            detail="Only models with a 'good' validation verdict can be published. "
            "Please recalibrate with a larger or higher-quality dataset to improve reliability.",
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

    validation_sweep_json = load_cached_validation_sweep(calibration_job_id)

    # Download model from calibration bucket
    try:
        model_bytes = download_from_bucket(
            BUCKET_CALIBRATION, f"{calibration_job_id}/model.pth"
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Model artifact not found")

    # Detect artifact type (try torchscript first)
    artifact_type = ArtifactType.PYTORCH.value
    try:
        torch.jit.load(io.BytesIO(model_bytes), map_location="cpu")
        artifact_type = ArtifactType.TORCHSCRIPT.value
    except Exception:
        pass

    # Upload model to published models bucket
    model_id = str(uuid.uuid4())
    artifact_path = f"{model_id}/model.pth"
    try:
        upload_to_bucket(BUCKET_MODELS, artifact_path, model_bytes)
    except Exception:
        raise HTTPException(status_code=503, detail="Failed to upload model artifact. Please try again.")

    # Copy config only if the job had one
    config_json_str = None
    if job.config_filename:
        try:
            config_bytes = download_from_bucket(
                BUCKET_CALIBRATION, f"{calibration_job_id}/config.json"
            )
            upload_to_bucket(BUCKET_MODELS, f"{model_id}/config.json", config_bytes, "application/json")
            config_json_str = config_bytes.decode()
        except Exception:
            raise HTTPException(status_code=503, detail="Failed to copy config artifact. Please try again.")

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
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        config_json=config_json_str,
        labels_json=json.dumps(labels),
        num_labels=len(labels),
        alpha=alpha,
        lamhat=lamhat,
        lamhat_result_json=job.result_json,
        validation_verdict=verdict,
        validation_metrics_json=json.dumps(metrics),
        validation_sweep_json=validation_sweep_json,
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


# ── Listing & Details ────────────────────────────────────────

def list_my_models(developer: User, db: Session) -> list[dict]:
    """List all published models owned by the developer."""
    models = (
        db.query(PublishedModel)
        .filter(PublishedModel.developer_id == developer.id)
        .order_by(PublishedModel.created_at.desc())
        .all()
    )
    return [model_to_summary(m, developer.full_name) for m in models]


def list_community_models(
    db: Session,
    search: str | None = None,
    modality: str | None = None,
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
    if sort == "alphabetical":
        q = q.order_by(PublishedModel.name.asc())
    else:
        q = q.order_by(PublishedModel.created_at.desc())

    results = q.all()
    return [model_to_summary(m, dev_name) for m, dev_name in results]


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
    return [model_to_summary(m, dev_name) for m, dev_name in results]


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
        if role == UserRole.CLINICIAN.value and vis not in (
            ModelVisibility.CLINICIAN.value,
            ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        if role == UserRole.DEVELOPER.value and vis not in (
            ModelVisibility.COMMUNITY.value,
            ModelVisibility.CLINICIAN_AND_COMMUNITY.value,
        ):
            raise HTTPException(status_code=403, detail="Access denied")

    return model_to_detail(model, dev_name)


# ── Updates & Visibility ─────────────────────────────────────

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
    expanding = is_expanding_visibility(old_vis, new_vis)
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


# ── Downloads ────────────────────────────────────────────────

def download_model_package(model_id: str, db: Session) -> tuple[bytes, str]:
    """Build a ZIP package with model.pth, config.json, lamhat.json, and validation_report.json."""
    result = (
        db.query(PublishedModel, User.full_name)
        .join(User, PublishedModel.developer_id == User.id)
        .filter(PublishedModel.id == model_id)
        .first()
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Published model not found")

    model, dev_name = result

    # Build the ZIP in memory
    buf = io.BytesIO()
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Stream the model artifact into the zip entry chunk-by-chunk.
            with zf.open("model.pth", "w") as model_entry:
                for chunk in iter_blob_chunks(BUCKET_MODELS, model.artifact_path):
                    model_entry.write(chunk)

            # config.json (preprocessing params)
            if model.config_json:
                zf.writestr("config.json", model.config_json)

            # lamhat.json (calibration results)
            if model.lamhat_result_json:
                zf.writestr("lamhat.json", model.lamhat_result_json)

            # validation_report.json
            validation_report = {
                "model_id": model.id,
                "model_name": model.name,
                "version": model.version,
                "developer": dev_name,
                "modality": model.modality,
                "intended_use": model.intended_use,
                "artifact_type": model.artifact_type,
                "labels": json.loads(model.labels_json),
                "num_labels": model.num_labels,
                "alpha": model.alpha,
                "lamhat": model.lamhat,
                "validation_verdict": model.validation_verdict,
                "validation_metrics": json.loads(model.validation_metrics_json) if model.validation_metrics_json else None,
                "published_at": model.created_at.isoformat() if model.created_at else None,
            }
            zf.writestr("validation_report.json", json.dumps(validation_report, indent=2))
    except Exception:
        raise HTTPException(status_code=404, detail="Model artifact file not found")

    filename = f"{model.name.replace(' ', '_')}_{model.version}.zip"
    return buf.getvalue(), filename


# ── Helpers ──────────────────────────────────────────────────

def load_cached_validation_sweep(calibration_job_id: str) -> str | None:
    """Load a cached validation sweep JSON string if one exists."""
    try:
        cached_bytes = download_from_bucket(
            BUCKET_CALIBRATION,
            result_path(calibration_job_id, VALIDATION_CACHE_FILENAME),
        )
    except Exception:
        return None

    try:
        cached = json.loads(cached_bytes)
        sweep = cached.get("sweep")
        return json.dumps(sweep) if sweep is not None else None
    except Exception:
        return None


def is_expanding_visibility(old: str, new: str) -> bool:
    """Return whether a visibility change expands the model audience."""
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


def update_model_details(
    model_id: str,
    description: str | None,
    intended_use: str | None,
    developer: User,
    db: Session,
) -> PublishedModel:
    """Update editable text fields on a published model."""
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

    if description is not None:
        model.description = description
    if intended_use is not None:
        model.intended_use = intended_use
    model.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(model)
    return model


def model_to_summary(model: PublishedModel, developer_name: str) -> dict:
    """Convert a published model row into summary API data."""
    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "version": model.version,
        "modality": model.modality,
        "intended_use": model.intended_use,
        "num_labels": model.num_labels,
        "labels": json.loads(model.labels_json) if model.labels_json else [],
        "alpha": model.alpha,
        "lamhat": model.lamhat,
        "validation_verdict": model.validation_verdict,
        "visibility": model.visibility,
        "is_active": model.is_active,
        "developer_name": developer_name,
        "created_at": model.created_at.isoformat() if model.created_at else None,
    }


def model_to_detail(model: PublishedModel, developer_name: str) -> dict:
    """Convert a published model row into detailed API data."""
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
