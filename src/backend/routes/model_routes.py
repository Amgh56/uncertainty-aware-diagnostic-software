"""Routes for Published Model management: publish, list, visibility, download."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from auth import get_current_user, require_developer
from database import get_db
from models import User
from schemas import (
    ErrorResponse,
    PublishModelRequest,
    ToggleActiveRequest,
    UpdateModelDetailsRequest,
    UpdateVisibilityRequest,
)
from services.published_model_service import (
    download_model_package,
    get_model_detail,
    list_clinician_models,
    list_community_models,
    list_my_models,
    publish_model,
    toggle_active,
    update_model_details,
    update_visibility,
)

router = APIRouter(tags=["Models"])


@router.post(
    "/models/publish",
    status_code=status.HTTP_201_CREATED,
    summary="Publish a calibration job as a model package",
    description=(
        "Create a Published Model Package from a completed, validated calibration job. "
        "Bundles the model artifact, config, labels, and calibration parameters."
    ),
    responses={
        201: {"description": "Model published successfully"},
        400: {"model": ErrorResponse, "description": "Validation error"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
def publish(
    body: PublishModelRequest,
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    model = publish_model(
        calibration_job_id=body.calibration_job_id,
        name=body.name,
        description=body.description,
        version=body.version,
        modality=body.modality,
        intended_use=body.intended_use,
        labels=body.labels,
        visibility=body.visibility,
        consent_agreed=body.consent_agreed,
        developer=developer,
        db=db,
    )
    return {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "visibility": model.visibility,
        "created_at": model.created_at.isoformat(),
    }


@router.get(
    "/models/mine",
    summary="List your published models",
    description="Returns all published models owned by the authenticated developer.",
    responses={
        200: {"description": "List of models"},
    },
)
def my_models(
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    return {"models": list_my_models(developer, db)}


@router.get(
    "/models/community",
    summary="Browse community models",
    description="List models shared by the developer community.",
    responses={
        200: {"description": "List of community models"},
    },
)
def community_models(
    search: str | None = None,
    modality: str | None = None,
    verdict: str | None = None,
    sort: str = "newest",
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    return {
        "models": list_community_models(db, search, modality, verdict, sort)
    }


@router.get(
    "/models/clinician",
    summary="List models available for clinical use",
    description="Returns models released for clinician inference.",
    responses={
        200: {"description": "List of clinician-available models"},
    },
)
def clinician_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return {"models": list_clinician_models(db)}


@router.get(
    "/models/{model_id}",
    summary="Get published model details",
    description="Returns full details of a published model with access control.",
    responses={
        200: {"description": "Model details"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
def model_detail(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_model_detail(model_id, current_user, db)


@router.patch(
    "/models/{model_id}/details",
    summary="Edit model description and intended use",
    description="Update the description and/or intended use of a published model. Only the owner can edit.",
    responses={
        200: {"description": "Model details updated"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
def edit_details(
    model_id: str,
    body: UpdateModelDetailsRequest,
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    model = update_model_details(model_id, body.description, body.intended_use, developer, db)
    return {
        "id": model.id,
        "description": model.description,
        "intended_use": model.intended_use,
    }


@router.patch(
    "/models/{model_id}/visibility",
    summary="Change model visibility",
    description="Update who can see and use this model. Expanding visibility requires consent.",
    responses={
        200: {"description": "Visibility updated"},
        400: {"model": ErrorResponse, "description": "Consent required"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
def change_visibility(
    model_id: str,
    body: UpdateVisibilityRequest,
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    model = update_visibility(model_id, body.visibility, body.consent_agreed, developer, db)
    return {"id": model.id, "visibility": model.visibility}


@router.patch(
    "/models/{model_id}/active",
    summary="Activate or deactivate a model",
    description="Deactivated models are hidden from clinicians and the community library.",
    responses={
        200: {"description": "Active status updated"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
def change_active(
    model_id: str,
    body: ToggleActiveRequest,
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    model = toggle_active(model_id, body.is_active, developer, db)
    return {"id": model.id, "is_active": model.is_active}


@router.get(
    "/models/{model_id}/download",
    summary="Download model package",
    description="Download a ZIP package containing model.pth, config.json, lamhat.json, and validation_report.json.",
    responses={
        200: {"description": "Model package ZIP download"},
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
def download_model(
    model_id: str,
    db: Session = Depends(get_db),
    developer: User = Depends(require_developer),
):
    data, filename = download_model_package(model_id, db)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
