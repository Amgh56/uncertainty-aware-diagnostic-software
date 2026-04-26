"""
Integration tests for the developer and model API endpoints.
Uses FastAPI TestClient with an in-memory SQLite DB (injected via conftest).
All Azure storage calls are mocked by the autouse mock_azure fixture in conftest.
"""

from unittest.mock import patch

import pytest

from tests.helpers import (
    make_clinician,
    make_dataset_zip,
    make_done_job,
    make_patient,
    make_published_model,
)
from enums import JobStatus, UncertaintyLevel, ValidationVerdict


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── POST /developer/jobs ──────────────────────────────────────────────────

@patch("routes.developer_routes.run_calibration_job")
def test_create_job_valid_upload(mock_run, client, dev_token, model_bytes, dataset_bytes):
    response = client.post(
        "/developer/jobs",
        params={"alpha": 0.1},
        headers=_auth(dev_token),
        files={
            "model_file": ("model.pt", model_bytes, "application/octet-stream"),
            "dataset_file": ("dataset.zip", dataset_bytes, "application/zip"),
        },
    )
    assert response.status_code == 201


@patch("routes.developer_routes.run_calibration_job")
def test_create_job_wrong_model_extension(mock_run, client, dev_token, dataset_bytes):
    """A .txt file as the model must be rejected immediately (extension validation)."""
    response = client.post(
        "/developer/jobs",
        params={"alpha": 0.1},
        headers=_auth(dev_token),
        files={
            "model_file": ("model.txt", b"not a model", "text/plain"),
            "dataset_file": ("dataset.zip", dataset_bytes, "application/zip"),
        },
    )
    assert response.status_code == 400


@patch("routes.developer_routes.run_calibration_job")
def test_create_job_missing_labels_csv(mock_run, client, dev_token, model_bytes):
    """
    The HTTP layer accepts the upload (201) because ZIP content is validated
    inside the background task, not at upload time.  We verify the service
    function raises a ValueError directly.
    """
    from services.calibration_service import extract_dataset_from_zip_bytes

    bad_zip = make_dataset_zip(n_images=60, include_labels_csv=False)
    with pytest.raises(ValueError, match="labels.csv"):
        extract_dataset_from_zip_bytes(bad_zip)


@patch("routes.developer_routes.run_calibration_job")
def test_create_job_alpha_out_of_range(mock_run, client, dev_token, model_bytes, dataset_bytes):
    # alpha is a query parameter on this endpoint, not a form field
    response = client.post(
        "/developer/jobs",
        params={"alpha": 1.5},
        headers=_auth(dev_token),
        files={
            "model_file": ("model.pt", model_bytes, "application/octet-stream"),
            "dataset_file": ("dataset.zip", dataset_bytes, "application/zip"),
        },
    )
    assert response.status_code == 400


# ── GET /developer/jobs/{job_id} ──────────────────────────────────────────

def test_get_job_not_owned(client, db, developer, other_developer, dev_token, other_dev_token):
    """A developer must not be able to retrieve another developer's job."""
    job = make_done_job(db, other_developer.id)
    response = client.get(f"/developer/jobs/{job.id}", headers=_auth(dev_token))
    assert response.status_code == 404


# ── GET /developer/jobs/{job_id}/validation ───────────────────────────────

def test_get_validation_job_not_complete(client, db, developer, dev_token):
    """Requesting validation for a queued (not completed) job must return 400."""
    import uuid
    from datetime import datetime, timezone
    from models import CalibrationJob

    job = CalibrationJob(
        id=str(uuid.uuid4()),
        display_name="Queued Job",
        developer_id=developer.id,
        status=JobStatus.QUEUED.value,
        model_filename="model.pt",
        dataset_filename="dataset.zip",
        alpha=0.1,
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()

    response = client.get(f"/developer/jobs/{job.id}/validation", headers=_auth(dev_token))
    assert response.status_code == 400


# ── POST /models/publish ──────────────────────────────────────────────────

def _publish_payload(job_id: str, visibility: str = "private", consent: bool = False) -> dict:
    return {
        "calibration_job_id": job_id,
        "name": "Test Model",
        "description": "A test model",
        "version": "1.0",
        "modality": "X-ray",
        "intended_use": "Testing",
        "labels": ["label_a", "label_b"],
        "visibility": visibility,
        "consent_agreed": consent,
    }


@patch("services.published_model_service.load_validation_artifacts", return_value=None)
def test_publish_without_good_verdict(mock_artifacts, client, db, developer, dev_token):
    """A job with verdict='review' must be rejected with 400."""
    job = make_done_job(db, developer.id, verdict=ValidationVerdict.REVIEW.value)
    response = client.post(
        "/models/publish",
        headers=_auth(dev_token),
        json=_publish_payload(job.id),
    )
    assert response.status_code == 400
    assert "good" in response.json()["detail"].lower()


# ── POST /predict — model routing ─────────────────────────────────────────

def test_inference_uses_selected_model(client, db, developer, dev_token):
    """
    When a clinician submits a prediction with model_id=A, the inference
    must run against model A — not model B or any cached model.
    Verified by:
      1. run_inference is called with the published_model whose id == model_a.id
      2. The response model_info.id matches model_a.id
      3. The response lamhat matches model_a's lamhat, not model_b's
    """
    from auth import create_access_token
    import cv2, numpy as np

    # Create two published models with distinct lamhats so we can tell them apart
    job_a = make_done_job(db, developer.id, verdict=ValidationVerdict.GOOD.value)
    job_b = make_done_job(db, developer.id, verdict=ValidationVerdict.GOOD.value)
    model_a = make_published_model(db, developer.id, job_a.id, lamhat=0.30, name="Model A")
    model_b = make_published_model(db, developer.id, job_b.id, lamhat=0.80, name="Model B")

    clinician = make_clinician(db)
    patient = make_patient(db, clinician.id)
    token = create_access_token(clinician.id)

    # Synthetic PNG image
    _, png = cv2.imencode(".png", np.zeros((32, 32), dtype=np.uint8))
    img_bytes = png.tobytes()

    # Fake findings that run_inference would return
    fake_findings = [
        {
            "finding": "label_a",
            "probability": 0.9,
            "uncertainty": UncertaintyLevel.LOW.value,
            "in_prediction_set": True,
        },
        {
            "finding": "label_b",
            "probability": 0.1,
            "uncertainty": UncertaintyLevel.HIGH.value,
            "in_prediction_set": False,
        },
    ]

    captured = {}

    def fake_run_inference(img, published_model):
        captured["model_id"] = published_model.id
        captured["lamhat"]   = published_model.lamhat
        return fake_findings

    with (
        patch("services.prediction_service.ml_state.run_inference", side_effect=fake_run_inference),
        patch("services.prediction_service.ml_state.upload_xray", return_value="http://fake/image.png"),
    ):
        response = client.post(
            "/predict",
            headers=_auth(token),
            files={"file": ("xray.png", img_bytes, "image/png")},
            data={"patient_id": patient.id, "model_id": model_a.id},
        )

    assert response.status_code == 200
    body = response.json()

    # run_inference was called with model_a, not model_b
    assert captured["model_id"] == model_a.id
    assert captured["model_id"] != model_b.id

    # Response carries model_a's identity and calibration parameters
    assert body["model_info"]["id"] == model_a.id
    assert body["lamhat"] == round(model_a.lamhat, 6)


@patch("services.published_model_service.load_validation_artifacts", return_value=None)
def test_publish_without_consent(mock_artifacts, client, db, developer, dev_token):
    """Non-private visibility without consent must be rejected with 400."""
    job = make_done_job(db, developer.id, verdict=ValidationVerdict.GOOD.value)
    response = client.post(
        "/models/publish",
        headers=_auth(dev_token),
        json=_publish_payload(job.id, visibility="community", consent=False),
    )
    assert response.status_code == 400
    assert "consent" in response.json()["detail"].lower()
