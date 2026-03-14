"""Business logic for doctor registration, login, and password reset."""

import hmac
import hashlib
import os
import time

from fastapi import HTTPException
from sqlalchemy.orm import Session

from auth import create_access_token, hash_password, verify_password
from enums import UserRole
from models import Doctor
from schemas import TokenResponse

_ENC_SECRET = os.getenv("ENC_SECRET_KEY", "change-me")
_TOKEN_TTL_SECONDS = 30 * 60  # 30 minutes


def _make_reset_token(email: str, timestamp: int) -> str:
    """HMAC-SHA256(email + timestamp, ENC_SECRET_KEY) as hex."""
    message = f"{email.lower().strip()}:{timestamp}".encode()
    return hmac.new(_ENC_SECRET.encode(), message, hashlib.sha256).hexdigest()


def generate_reset_token(email: str) -> tuple[str, int]:
    """Return (token, timestamp) for embedding in the reset link."""
    ts = int(time.time())
    return _make_reset_token(email, ts), ts


def verify_reset_token(email: str, token: str, timestamp: int) -> bool:
    """Return True if token is valid and not expired."""
    if int(time.time()) - timestamp > _TOKEN_TTL_SECONDS:
        return False
    expected = _make_reset_token(email, timestamp)
    return hmac.compare_digest(expected, token)


def register_doctor(
    email: str,
    password: str,
    full_name: str,
    db: Session,
    role: UserRole = UserRole.CLINICIAN,
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
        role=role.value,
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


def forgot_password(email: str, db: Session) -> dict:
    """Generate a reset token and return the link. Always returns 200 to avoid email enumeration."""
    doctor = db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    if not doctor:
        # Return success anyway — don't reveal whether email exists
        return {"email": email, "token": None, "timestamp": None}

    token, timestamp = generate_reset_token(email)
    return {"email": email.lower().strip(), "token": token, "timestamp": timestamp}


def reset_password(email: str, token: str, timestamp: int, new_password: str, db: Session) -> dict:
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    if not verify_reset_token(email, token, timestamp):
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    doctor = db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    if not doctor:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    doctor.hashed_password = hash_password(new_password)
    db.commit()
    return {"detail": "Password updated successfully"}
