"""Business logic for patient management."""

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import User, Patient, Prediction
from schemas import PatientListItem, PatientListResponse


def create_or_get_patient(
    mrn: str, first_name: str, last_name: str, user: User, db: Session
) -> Patient:
    existing = (
        db.query(Patient)
        .filter(Patient.mrn == mrn.strip(), Patient.user_id == user.id)
        .first()
    )
    if existing:
        return existing

    patient = Patient(
        mrn=mrn.strip(),
        first_name=first_name.strip(),
        last_name=last_name.strip(),
        user_id=user.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def list_patients_with_stats(user: User, db: Session) -> PatientListResponse:
    # Single query: join patients with aggregated prediction stats
    last_pred = (
        db.query(
            Prediction.patient_id,
            func.count(Prediction.id).label("pred_count"),
            func.max(Prediction.created_at).label("last_at"),
        )
        .filter(Prediction.user_id == user.id)
        .group_by(Prediction.patient_id)
        .subquery()
    )

    rows = (
        db.query(Patient, last_pred.c.pred_count, last_pred.c.last_at)
        .outerjoin(last_pred, Patient.id == last_pred.c.patient_id)
        .filter(Patient.user_id == user.id)
        .order_by(Patient.created_at.desc())
        .all()
    )

    # Batch-fetch the latest prediction per patient (for top_finding + id)
    patient_ids = [pat.id for pat, _, last_at in rows if last_at is not None]
    latest_preds = {}
    if patient_ids:
        for pred in (
            db.query(Prediction)
            .filter(
                Prediction.patient_id.in_(patient_ids),
                Prediction.user_id == user.id,
            )
            .order_by(Prediction.created_at.desc())
            .all()
        ):
            if pred.patient_id not in latest_preds:
                latest_preds[pred.patient_id] = pred

    items = []
    for pat, pred_count, last_at in rows:
        lp = latest_preds.get(pat.id)
        items.append(
            PatientListItem(
                id=pat.id,
                mrn=pat.mrn,
                first_name=pat.first_name,
                last_name=pat.last_name,
                prediction_count=pred_count or 0,
                last_prediction_at=lp.created_at if lp else None,
                last_top_finding=lp.top_finding if lp else None,
                last_prediction_id=lp.id if lp else None,
                created_at=pat.created_at,
            )
        )

    return PatientListResponse(patients=items)


def get_patient_predictions(patient_id: int, user: User, db: Session) -> list[dict]:
    """Return all predictions for a given patient, newest first."""
    from models import PublishedModel
    import json

    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.user_id == user.id)
        .first()
    )
    if not patient:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Patient not found")

    predictions = (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id, Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    items = []
    for pred in predictions:
        findings = json.loads(pred.findings_json)
        model_info = None
        if pred.published_model_id:
            pub_model = (
                db.query(PublishedModel)
                .filter(PublishedModel.id == pred.published_model_id)
                .first()
            )
            if pub_model:
                model_info = {
                    "id": pub_model.id,
                    "name": pub_model.name,
                    "version": pub_model.version,
                    "modality": pub_model.modality,
                    "num_labels": pub_model.num_labels,
                    "validation_verdict": pub_model.validation_verdict,
                }
        items.append({
            "id": pred.id,
            "patient_id": pred.patient_id,
            "image_path": pred.image_path,
            "top_finding": pred.top_finding,
            "top_probability": pred.top_probability,
            "prediction_set_size": pred.prediction_set_size,
            "coverage": pred.coverage,
            "alpha": pred.alpha,
            "lamhat": pred.lamhat,
            "findings": findings,
            "created_at": pred.created_at.isoformat(),
            "model_info": model_info,
        })

    return {
        "patient": {
            "id": patient.id,
            "mrn": patient.mrn,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "created_at": patient.created_at.isoformat(),
        },
        "predictions": items,
    }
