from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
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
from enums import JobStatus, ModelVisibility, UserRole


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(
        String(20),
        nullable=False,
        default=UserRole.CLINICIAN.value,
        server_default=UserRole.CLINICIAN.value,
    )
    is_verified = Column(Boolean, nullable=False, default=False, server_default="0")
    otp_code = Column(String(64), nullable=True)
    otp_expires_at = Column(DateTime, nullable=True)
    otp_attempts = Column(Integer, nullable=False, default=0, server_default="0")
    otp_last_sent_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # one to many 
    patients = relationship("Patient", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")
    calibration_jobs = relationship("CalibrationJob", back_populates="developer")
    published_models = relationship("PublishedModel", back_populates="developer")


class Patient(Base):
    __tablename__ = "patients"
    __table_args__ = (
        UniqueConstraint("mrn", "user_id", name="uq_patient_mrn_user"),
        Index("ix_patient_user_id", "user_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    mrn = Column(String(100), nullable=False)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # many to one 
    user = relationship("User", back_populates="patients")
    predictions = relationship(
        "Prediction", back_populates="patient", order_by="desc(Prediction.created_at)"
    )


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_prediction_user_created", "user_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    top_finding = Column(String(100), nullable=False)
    top_probability = Column(Float, nullable=False)
    prediction_set_size = Column(Integer, nullable=False)
    coverage = Column(String(10), nullable=False)
    alpha = Column(Float, nullable=False)
    lamhat = Column(Float, nullable=False)
    findings_json = Column(Text, nullable=False)
    published_model_id = Column(
        String(36), ForeignKey("published_models.id"), nullable=True
    )
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    patient = relationship("Patient", back_populates="predictions")
    user = relationship("User", back_populates="predictions")
    published_model = relationship("PublishedModel", back_populates="predictions")


class CalibrationJob(Base):
    __tablename__ = "calibration_jobs"
    __table_args__ = (
        Index("ix_calib_job_developer", "developer_id"),
    )

    id = Column(String(36), primary_key=True)
    developer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(20), nullable=False, default=JobStatus.QUEUED.value)
    model_filename = Column(String(255), nullable=False)
    config_filename = Column(String(255), nullable=True)
    dataset_filename = Column(String(255), nullable=False)
    alpha = Column(Float, nullable=False, default=0.1)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    validation_verdict = Column(String(20), nullable=True)
    is_published = Column(Boolean, nullable=False, default=False, server_default="0")
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    completed_at = Column(DateTime, nullable=True)

    developer = relationship("User", back_populates="calibration_jobs")
    published_model = relationship(
        "PublishedModel", back_populates="calibration_job", uselist=False
    )


class PublishedModel(Base):
    __tablename__ = "published_models"
    __table_args__ = (
        Index("ix_pub_model_visibility_active", "visibility", "is_active"),
        Index("ix_pub_model_developer", "developer_id"),
        Index("ix_pub_model_modality", "modality"),
    )

    id = Column(String(36), primary_key=True)
    calibration_job_id = Column(
        String(36), ForeignKey("calibration_jobs.id"), unique=True, nullable=False
    )
    developer_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Identity
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=False)
    version = Column(String(20), nullable=False)

    # Classification
    modality = Column(String(100), nullable=False)
    intended_use = Column(Text, nullable=False)

    # Technical package
    artifact_path = Column(String(500), nullable=False)
    artifact_type = Column(String(20), nullable=False, default="pytorch")
    config_json = Column(Text, nullable=True)
    labels_json = Column(Text, nullable=False)
    num_labels = Column(Integer, nullable=False)

    # Calibration outputs
    alpha = Column(Float, nullable=False)
    lamhat = Column(Float, nullable=False)
    lamhat_result_json = Column(Text, nullable=True)

    # Validation outputs
    validation_verdict = Column(String(20), nullable=False)
    validation_metrics_json = Column(Text, nullable=True)

    # Visibility & release
    visibility = Column(
        String(30),
        nullable=False,
        default=ModelVisibility.PRIVATE.value,
    )
    is_active = Column(Boolean, nullable=False, default=True, server_default="1")

    # Consent
    consent_given_at = Column(DateTime, nullable=True)
    consent_text_hash = Column(String(64), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    developer = relationship("User", back_populates="published_models")
    calibration_job = relationship("CalibrationJob", back_populates="published_model")
    predictions = relationship("Prediction", back_populates="published_model")
