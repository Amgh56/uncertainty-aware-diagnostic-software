"""Business logic for doctor registration and login."""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from auth import create_access_token, hash_password, verify_password
from models import Doctor
from schemas import TokenResponse


def register_doctor(
    email: str, password: str, full_name: str, db: Session
) -> TokenResponse:
    if len(password) < 6:
        raise HTTPException(
            status_code=400, detail="Password must be at least 6 characters"
        )

    existing = (
        db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    doctor = Doctor(
        email=email.lower().strip(),
        hashed_password=hash_password(password),
        full_name=full_name.strip(),
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    token = create_access_token(doctor.id)
    return TokenResponse(access_token=token)


def login_doctor(email: str, password: str, db: Session) -> TokenResponse:
    doctor = (
        db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    )
    if not doctor or not verify_password(password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(doctor.id)
    return TokenResponse(access_token=token)
