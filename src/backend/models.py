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


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    patients = relationship("Patient", back_populates="doctor")
    predictions = relationship("Prediction", back_populates="doctor")


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
