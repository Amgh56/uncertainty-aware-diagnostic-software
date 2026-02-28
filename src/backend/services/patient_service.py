"""Business logic for patient management."""

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import Doctor, Patient, Prediction
from schemas import PatientListItem, PatientListResponse


def create_or_get_patient(
    mrn: str, first_name: str, last_name: str, doctor: Doctor, db: Session
) -> Patient:
    existing = (
        db.query(Patient)
        .filter(Patient.mrn == mrn.strip(), Patient.doctor_id == doctor.id)
        .first()
    )
    if existing:
        return existing

    patient = Patient(
        mrn=mrn.strip(),
        first_name=first_name.strip(),
        last_name=last_name.strip(),
        doctor_id=doctor.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def list_patients_with_stats(doctor: Doctor, db: Session) -> PatientListResponse:
    patients = (
        db.query(Patient)
        .filter(Patient.doctor_id == doctor.id)
        .order_by(Patient.created_at.desc())
        .all()
    )

    items = []
    for pat in patients:
        last_pred = (
            db.query(Prediction)
            .filter(Prediction.patient_id == pat.id)
            .order_by(Prediction.created_at.desc())
            .first()
        )
        pred_count = (
            db.query(func.count(Prediction.id))
            .filter(Prediction.patient_id == pat.id)
            .scalar()
        )
        items.append(
            PatientListItem(
                id=pat.id,
                mrn=pat.mrn,
                first_name=pat.first_name,
                last_name=pat.last_name,
                prediction_count=pred_count,
                last_prediction_at=last_pred.created_at if last_pred else None,
                last_top_finding=last_pred.top_finding if last_pred else None,
                last_prediction_id=last_pred.id if last_pred else None,
                created_at=pat.created_at,
            )
        )

    return PatientListResponse(patients=items)
