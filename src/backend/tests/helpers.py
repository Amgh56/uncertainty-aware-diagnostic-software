"""
Shared synthetic-data builders used by conftest.py and test files.
No app imports here — this module must be safe to import standalone.
"""

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import torch


def make_torchscript_bytes() -> bytes:
    """Minimal TorchScript identity model serialised to bytes."""
    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    traced = torch.jit.trace(_Identity(), torch.zeros(1, 3, 4, 4))
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def make_dataset_zip(
    n_images: int = 60,
    include_labels_csv: bool = True,
    unsafe_path: bool = False,
) -> bytes:
    """Return a minimal dataset zip in bytes."""
    rng = np.random.default_rng(0)
    filenames = [f"img_{i:04d}.png" for i in range(n_images)]
    blank = np.zeros((32, 32), dtype=np.uint8)
    _, png_bytes = cv2.imencode(".png", blank)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if unsafe_path:
            zf.writestr("../evil.txt", "bad content")

        if include_labels_csv:
            rows = [
                {
                    "filename": f,
                    "label_a": int(rng.integers(0, 2)),
                    "label_b": int(rng.integers(0, 2)),
                }
                for f in filenames
            ]
            rows[0]["label_a"] = 1  # guarantee at least one positive row
            df = pd.DataFrame(rows)
            zf.writestr("labels.csv", df.to_csv(index=False))

        for fname in filenames:
            zf.writestr(f"images/{fname}", png_bytes.tobytes())

    return buf.getvalue()


def make_clinician(db, email: str = "clinician@test.com"):
    """Insert a verified clinician user into the test DB."""
    from auth import hash_password
    from enums import UserRole
    from models import User

    user = User(
        email=email,
        hashed_password=hash_password("password123"),
        full_name="Test Clinician",
        role=UserRole.CLINICIAN.value,
        is_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def make_patient(db, user_id: int, mrn: str = "MRN001"):
    """Insert a patient belonging to a user into the test DB."""
    from models import Patient

    patient = Patient(
        mrn=mrn,
        first_name="John",
        last_name="Doe",
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def make_published_model(db, developer_id: int, job_id: str, lamhat: float = 0.42, name: str = "Model A"):
    """Insert an active published model into the test DB."""
    from enums import ModelVisibility
    from models import PublishedModel

    model = PublishedModel(
        id=str(uuid.uuid4()),
        calibration_job_id=job_id,
        developer_id=developer_id,
        name=name,
        description="Test model",
        version="1.0",
        modality="X-ray",
        intended_use="Testing",
        artifact_path=f"{str(uuid.uuid4())}/model.pth",
        artifact_type="pytorch",
        labels_json=json.dumps(["label_a", "label_b"]),
        num_labels=2,
        alpha=0.1,
        lamhat=lamhat,
        validation_verdict="good",
        visibility=ModelVisibility.CLINICIAN.value,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def make_done_job(db, developer_id: int, verdict: str = "good"):
    """Insert a completed CalibrationJob directly into the test DB."""
    from enums import JobStatus
    from models import CalibrationJob

    job = CalibrationJob(
        id=str(uuid.uuid4()),
        display_name="Test Job",
        developer_id=developer_id,
        status=JobStatus.DONE.value,
        model_filename="model.pt",
        dataset_filename="dataset.zip",
        alpha=0.1,
        validation_verdict=verdict,
        result_json=json.dumps({
            "lamhat": 0.42,
            "alpha": 0.1,
            "n_samples": 60,
            "labels": ["label_a", "label_b"],
        }),
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job
