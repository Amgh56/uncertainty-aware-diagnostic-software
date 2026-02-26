"""
FastAPI backend for the Uncertainty-Aware Diagnostic System.

Runs single-image conformal prediction inference using the pretrained CheXpert model
and calibrated lamhat loaded once at server startup.

Usage:
    cd src/backend
    pip install -r requirements-api.txt
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
import json
import json as json_module
import os
import time

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from conformal_prediction_pipeline import (
    DISEASES,
    load_chexpert_pretrained_model,
    pick_device,
    predict_batch_probs,
    preprocess_image_from_bytes,
)
from database import Base, engine, get_db
from models import Doctor, Patient, Prediction
from schemas import (
    DoctorLoginRequest,
    DoctorRegisterRequest,
    DoctorResponse,
    HistoryItem,
    HistoryResponse,
    PatientCreateRequest,
    PatientListItem,
    PatientListResponse,
    PatientResponse,
    TokenResponse,
)
from auth import (
    create_access_token,
    get_current_doctor,
    hash_password,
    verify_password,
)


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent


app = FastAPI(
    title="Uncertainty-Aware Chest X-ray Diagnostic API",
    description="Conformal prediction sets for multi-label chest X-ray classification",
    version="1.0.0",
)

cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # Allow localhost/127.0.0.1 on arbitrary dev ports (Vite may switch ports).
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
config = None
device = None
lamhat = None
alpha = None


def is_supported_upload(file: UploadFile) -> bool:
    if not file:
        return False

    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    if content_type in {"image/png", "image/jpeg"}:
        return True

    if filename.endswith((".png", ".jpg", ".jpeg")):
        return True

    return False


def load_lamhat_json() -> tuple[float, float, Path]:
    lamhat_path = BACKEND_DIR / "NIH_dataset" / "artifacts" / "lamhat.json"

    if not lamhat_path.exists():
        raise FileNotFoundError(
            f"lamhat.json not found at: {lamhat_path}. Run calibration first."
        )

    with open(lamhat_path, "r") as f:
        payload = json.load(f)

    if "lamhat" not in payload or "alpha" not in payload:
        raise ValueError(f"lamhat payload missing keys in {lamhat_path}")

    return float(payload["lamhat"]), float(payload["alpha"]), lamhat_path


@app.on_event("startup")
def startup() -> None:
    global model, config, device, lamhat, alpha

    # Create DB tables
    Base.metadata.create_all(bind=engine)

    # Create uploads directory
    uploads_dir = BACKEND_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    model, config = load_chexpert_pretrained_model()
    device = pick_device()
    model = model.to(device)

    lamhat, alpha, lamhat_path = load_lamhat_json()

    print(f"Model loaded on device: {device}")
    print(f"Loaded lamhat={lamhat:.6f}, alpha={alpha} from {lamhat_path}")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/register", response_model=TokenResponse)
def register(body: DoctorRegisterRequest, db: Session = Depends(get_db)):
    if len(body.password) < 6:
        raise HTTPException(
            status_code=400, detail="Password must be at least 6 characters"
        )

    existing = (
        db.query(Doctor).filter(Doctor.email == body.email.lower().strip()).first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    doctor = Doctor(
        email=body.email.lower().strip(),
        hashed_password=hash_password(body.password),
        full_name=body.full_name.strip(),
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    token = create_access_token(doctor.id)
    return TokenResponse(access_token=token)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: DoctorLoginRequest, db: Session = Depends(get_db)):
    doctor = (
        db.query(Doctor).filter(Doctor.email == body.email.lower().strip()).first()
    )
    if not doctor or not verify_password(body.password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(doctor.id)
    return TokenResponse(access_token=token)


@app.get("/auth/me", response_model=DoctorResponse)
def get_me(current_doctor: Doctor = Depends(get_current_doctor)):
    return current_doctor


# ---------------------------------------------------------------------------
# Patient endpoints
# ---------------------------------------------------------------------------


@app.post("/patients", response_model=PatientResponse, status_code=201)
def create_or_get_patient(
    body: PatientCreateRequest,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    existing = (
        db.query(Patient)
        .filter(
            Patient.mrn == body.mrn.strip(),
            Patient.doctor_id == current_doctor.id,
        )
        .first()
    )

    if existing:
        return existing

    patient = Patient(
        mrn=body.mrn.strip(),
        first_name=body.first_name.strip(),
        last_name=body.last_name.strip(),
        doctor_id=current_doctor.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@app.get("/patients", response_model=PatientListResponse)
def list_patients(
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    from sqlalchemy import func

    patients = (
        db.query(Patient)
        .filter(Patient.doctor_id == current_doctor.id)
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


# ---------------------------------------------------------------------------
# Predict endpoint (modified — now requires auth + patient_id)
# ---------------------------------------------------------------------------


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: int = Form(...),
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    if not is_supported_upload(file):
        raise HTTPException(status_code=400, detail="File must be PNG, JPEG")

    # Verify patient belongs to this doctor
    patient = (
        db.query(Patient)
        .filter(
            Patient.id == patient_id,
            Patient.doctor_id == current_doctor.id,
        )
        .first()
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    try:
        img_bytes = await file.read()
        if not img_bytes:
            raise ValueError("Uploaded file is empty")

        # Save image to disk
        doctor_upload_dir = BACKEND_DIR / "uploads" / str(current_doctor.id)
        doctor_upload_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        safe_filename = f"{timestamp}_{file.filename}"
        image_disk_path = doctor_upload_dir / safe_filename
        with open(image_disk_path, "wb") as f:
            f.write(img_bytes)

        relative_image_path = f"uploads/{current_doctor.id}/{safe_filename}"

        # Run ML inference
        img_array = preprocess_image_from_bytes(img_bytes, config)

        x = torch.tensor(
            img_array[np.newaxis],
            dtype=torch.float32,
            device=device,
        )

        probs = predict_batch_probs(model, x)[0]

        findings = []
        for i, disease in enumerate(DISEASES):
            p = float(probs[i])
            in_set = bool(p >= lamhat)

            if p >= 0.7:
                uncertainty = "Low"
            elif p >= 0.4:
                uncertainty = "Medium"
            else:
                uncertainty = "High"

            findings.append(
                {
                    "finding": disease,
                    "probability": round(p, 4),
                    "uncertainty": uncertainty,
                    "in_prediction_set": in_set,
                }
            )

        findings.sort(key=lambda row: row["probability"], reverse=True)

        prediction_set_size = sum(1 for row in findings if row["in_prediction_set"])
        top = findings[0]

        # Save prediction to database
        prediction = Prediction(
            patient_id=patient.id,
            doctor_id=current_doctor.id,
            image_path=relative_image_path,
            top_finding=top["finding"],
            top_probability=top["probability"],
            prediction_set_size=prediction_set_size,
            coverage=f"{int((1 - alpha) * 100)}%",
            alpha=alpha,
            lamhat=round(lamhat, 6),
            findings_json=json_module.dumps(findings),
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        return {
            "id": prediction.id,
            "patient_id": patient.id,
            "image_path": relative_image_path,
            "top_finding": top["finding"],
            "top_probability": top["probability"],
            "prediction_set_size": prediction_set_size,
            "coverage": f"{int((1 - alpha) * 100)}%",
            "alpha": alpha,
            "lamhat": round(lamhat, 6),
            "findings": findings,
            "created_at": prediction.created_at.isoformat(),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


# ---------------------------------------------------------------------------
# History & prediction detail endpoints
# ---------------------------------------------------------------------------


@app.get("/history", response_model=HistoryResponse)
def get_history(
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Prediction, Patient)
        .join(Patient, Prediction.patient_id == Patient.id)
        .filter(Prediction.doctor_id == current_doctor.id)
        .order_by(Prediction.created_at.desc())
        .limit(10)
        .all()
    )

    items = []
    for pred, pat in rows:
        items.append(
            HistoryItem(
                prediction_id=pred.id,
                patient_id=pat.id,
                mrn=pat.mrn,
                first_name=pat.first_name,
                last_name=pat.last_name,
                top_finding=pred.top_finding,
                top_probability=pred.top_probability,
                prediction_set_size=pred.prediction_set_size,
                created_at=pred.created_at,
            )
        )

    return HistoryResponse(items=items)


@app.get("/predictions/{prediction_id}")
def get_prediction(
    prediction_id: int,
    current_doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    prediction = (
        db.query(Prediction)
        .filter(
            Prediction.id == prediction_id,
            Prediction.doctor_id == current_doctor.id,
        )
        .first()
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    patient = db.query(Patient).filter(Patient.id == prediction.patient_id).first()
    findings = json_module.loads(prediction.findings_json)

    return {
        "id": prediction.id,
        "patient_id": prediction.patient_id,
        "image_path": prediction.image_path,
        "top_finding": prediction.top_finding,
        "top_probability": prediction.top_probability,
        "prediction_set_size": prediction.prediction_set_size,
        "coverage": prediction.coverage,
        "alpha": prediction.alpha,
        "lamhat": prediction.lamhat,
        "findings": findings,
        "created_at": prediction.created_at.isoformat(),
        "patient": {
            "id": patient.id,
            "mrn": patient.mrn,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "created_at": patient.created_at.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Serve uploaded images
# ---------------------------------------------------------------------------


@app.get("/uploads/{file_path:path}")
async def serve_upload(file_path: str):
    full_path = BACKEND_DIR / "uploads" / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(full_path))


# ---------------------------------------------------------------------------
# Health check (public, no auth)
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "lamhat": lamhat,
        "alpha": alpha,
        "diseases": DISEASES,
    }
