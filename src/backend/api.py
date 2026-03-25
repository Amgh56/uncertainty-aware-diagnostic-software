"""
FastAPI backend for the Uncertainty-Aware Diagnostic System.

Uses published models with conformal prediction for uncertainty-aware
multi-label medical image classification.

Usage:
    cd src/backend
    pip install -r requirements-api.txt
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine

from azure_client import ensure_buckets
from routes.auth_routes import router as auth_router
from routes.patient_routes import router as patient_router
from routes.prediction_routes import router as prediction_router
from routes.developer_routes import router as developer_router
from routes.model_routes import router as model_router


tags_metadata = [
    {
        "name": "Auth",
        "description": "User registration, login, and profile endpoints. "
        "All authenticated endpoints require a Bearer token in the Authorization header.",
    },
    {
        "name": "Patients",
        "description": "Create and list patients. Each patient is scoped to the authenticated user.",
    },
    {
        "name": "Predictions",
        "description": "Upload medical images for ML inference using published models, "
        "view prediction history and details. "
        "Uses conformal prediction for uncertainty-aware multi-label classification.",
    },
    {
        "name": "Developer",
        "description": (
            "Developer / Researcher mode. Upload a pretrained model (.pth/.pt) "
            "and a labelled calibration dataset (.zip) to run the conformal calibration pipeline "
            "in the background. Download a `lamhat.json` with the calibrated threshold and metrics. "
            "Requires a **developer** role account (`POST /developer/register`)."
        ),
    },
    {
        "name": "Models",
        "description": (
            "Published Model Packages. Developers publish calibrated models; "
            "clinicians browse and use them for inference; the community can download and reuse them."
        ),
    },
]

app = FastAPI(
    title="Uncertainty-Aware Diagnostic API",
    description=(
        "REST API for uncertainty-aware medical image diagnosis.\n\n"
        "Developers upload and calibrate models using **conformal prediction**. "
        "Clinicians use published models to get calibrated prediction sets "
        "with uncertainty quantification.\n\n"
        "**Authentication:** Most endpoints require a JWT Bearer token obtained via `/auth/login` or `/auth/register`."
    ),
    version="2.0.0",
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
app.include_router(developer_router)
app.include_router(model_router)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_buckets()
