from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from database import Base


# ---------------------------------------------------------------------------
# Roles & Job status (plain strings — no DB Enum type for portability)
# ---------------------------------------------------------------------------

class UserRole:
    CLINICIAN = "clinician"
    DEVELOPER = "developer"


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Existing models
# ---------------------------------------------------------------------------

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.CLINICIAN,
                  server_default=UserRole.CLINICIAN)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    patients = relationship("Patient", back_populates="doctor")
    predictions = relationship("Prediction", back_populates="doctor")
    calibration_jobs = relationship("CalibrationJob", back_populates="developer")


class Patient(Base):
    __tablename__ = "patients"
    __table_args__ = (
        UniqueConstraint("mrn", "doctor_id", name="uq_patient_mrn_doctor"),
        Index("ix_patient_doctor_id", "doctor_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    mrn = Column(String(100), nullable=False)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    doctor = relationship("Doctor", back_populates="patients")
    predictions = relationship(
        "Prediction", back_populates="patient", order_by="desc(Prediction.created_at)"
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_prediction_doctor_created", "doctor_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    top_finding = Column(String(100), nullable=False)
    top_probability = Column(Float, nullable=False)
    prediction_set_size = Column(Integer, nullable=False)
    coverage = Column(String(10), nullable=False)
    alpha = Column(Float, nullable=False)
    lamhat = Column(Float, nullable=False)
    findings_json = Column(Text, nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    patient = relationship("Patient", back_populates="predictions")
    doctor = relationship("Doctor", back_populates="predictions")


# ---------------------------------------------------------------------------
# Developer / Researcher models
# ---------------------------------------------------------------------------

class CalibrationJob(Base):
    __tablename__ = "calibration_jobs"
    __table_args__ = (
        Index("ix_calib_job_developer", "developer_id"),
    )

    id = Column(String(36), primary_key=True)           # UUID string
    developer_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    status = Column(String(20), nullable=False, default=JobStatus.QUEUED)
    model_filename = Column(String(255), nullable=False)
    config_filename = Column(String(255), nullable=True)   # uploaded model config JSON
    dataset_filename = Column(String(255), nullable=False)
    alpha = Column(Float, nullable=False, default=0.1)
    result_json = Column(Text, nullable=True)            # lamhat + metrics JSON
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    completed_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=False)        # TTL for cleanup

    developer = relationship("Doctor", back_populates="calibration_jobs")
