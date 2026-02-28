"""
FastAPI backend for the Uncertainty-Aware Diagnostic System.

Runs single-image conformal prediction inference using the pretrained CheXpert model
and calibrated lamhat loaded once at server startup.

Usage:
    cd src/backend
    pip install -r requirements-api.txt
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
from schemas import HealthResponse
from supabase_client import ensure_bucket
from services.ml_service import ml_state
from routes.auth_routes import router as auth_router
from routes.patient_routes import router as patient_router
from routes.prediction_routes import router as prediction_router


tags_metadata = [
    {
        "name": "Auth",
        "description": "Doctor registration, login, and profile endpoints. "
        "All authenticated endpoints require a Bearer token in the Authorization header.",
    },
    {
        "name": "Patients",
        "description": "Create and list patients. Each patient is scoped to the authenticated doctor.",
    },
    {
        "name": "Predictions",
        "description": "Upload chest X-ray images for ML inference, view prediction history and details. "
        "Uses conformal prediction for uncertainty-aware multi-label classification of 5 diseases.",
    },
    {
        "name": "Health",
        "description": "Server health check (public, no authentication required).",
    },
]

app = FastAPI(
    title="Uncertainty-Aware Chest X-ray Diagnostic API",
    description=(
        "REST API for uncertainty-aware chest X-ray diagnosis.\n\n"
        "Uses a pretrained CheXpert model with **conformal prediction** to provide "
        "calibrated prediction sets for 5 diseases: "
        "Cardiomegaly, Edema, Consolidation, Atelectasis, and Pleural Effusion.\n\n"
        "**Authentication:** Most endpoints require a JWT Bearer token obtained via `/auth/login` or `/auth/register`."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(prediction_router)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_bucket()
    ml_state.load()


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Public endpoint that returns the server status, "
    "whether the ML model is loaded, the device in use, "
    "and the calibration parameters (lamhat, alpha).",
)
def health():
    return {
        "status": "ok",
        "model_loaded": ml_state.model is not None,
        "device": ml_state.device,
        "lamhat": ml_state.lamhat,
        "alpha": ml_state.alpha,
        "diseases": ml_state.diseases,
    }
