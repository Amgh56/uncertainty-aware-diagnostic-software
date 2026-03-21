"""Business logic for doctor registration, login, password reset, and email OTP verification."""

import hashlib
import hmac
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from auth import create_access_token, hash_password, verify_password
from enums import UserRole
from models import Doctor
from schemas import TokenResponse

_ENC_SECRET = os.getenv("ENC_SECRET_KEY", "change-me")
_TOKEN_TTL_SECONDS = 30 * 60  # 30 minutes
_OTP_TTL_SECONDS = 10 * 60  # 10 minutes
_OTP_MAX_ATTEMPTS = 5
_OTP_RESEND_COOLDOWN_SECONDS = 60


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


def _generate_otp() -> str:
    """Generate a random 6-digit OTP."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_otp(otp: str) -> str:
    """Hash OTP with SHA-256 for secure storage."""
    return hashlib.sha256(otp.encode()).hexdigest()


def register_doctor(
    email: str,
    password: str,
    full_name: str,
    db: Session,
    role: UserRole = UserRole.CLINICIAN,
) -> dict:
    if len(password) < 6:
        raise HTTPException(
            status_code=400, detail="Password must be at least 6 characters"
        )

    existing = (
        db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    otp = _generate_otp()
    now = datetime.now(timezone.utc)

    doctor = Doctor(
        email=email.lower().strip(),
        hashed_password=hash_password(password),
        full_name=full_name.strip(),
        role=role.value,
        is_verified=False,
        otp_code=_hash_otp(otp),
        otp_expires_at=now + timedelta(seconds=_OTP_TTL_SECONDS),
        otp_attempts=0,
        otp_last_sent_at=now,
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    token = create_access_token(doctor.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "is_verified": False,
        "otp": otp,
        "email": doctor.email,
        "full_name": doctor.full_name,
    }


def login_doctor(email: str, password: str, db: Session) -> dict:
    doctor = (
        db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    )
    if not doctor or not verify_password(password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(doctor.id)

    # If unverified, generate a fresh OTP so the email can be sent
    otp = None
    if not doctor.is_verified:
        otp = _generate_otp()
        now = datetime.now(timezone.utc)
        doctor.otp_code = _hash_otp(otp)
        doctor.otp_expires_at = now + timedelta(seconds=_OTP_TTL_SECONDS)
        doctor.otp_attempts = 0
        doctor.otp_last_sent_at = now
        db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
        "is_verified": doctor.is_verified,
        "otp": otp,
        "email": doctor.email,
        "full_name": doctor.full_name,
    }


def verify_email_otp(email: str, otp: str, db: Session) -> dict:
    doctor = db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    if not doctor:
        raise HTTPException(status_code=400, detail="Invalid email")

    if doctor.is_verified:
        return {"detail": "Email already verified"}

    if doctor.otp_attempts >= _OTP_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please request a new code.",
        )

    if not doctor.otp_code or not doctor.otp_expires_at:
        raise HTTPException(
            status_code=400,
            detail="No verification code found. Please request a new code.",
        )

    now = datetime.now(timezone.utc)
    expires = doctor.otp_expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if now > expires:
        raise HTTPException(
            status_code=400,
            detail="Verification code has expired. Please request a new code.",
        )

    if not hmac.compare_digest(_hash_otp(otp), doctor.otp_code):
        doctor.otp_attempts += 1
        db.commit()
        remaining = _OTP_MAX_ATTEMPTS - doctor.otp_attempts
        raise HTTPException(
            status_code=400,
            detail=f"Incorrect code. {remaining} attempt{'s' if remaining != 1 else ''} remaining.",
        )

    doctor.is_verified = True
    doctor.otp_code = None
    doctor.otp_expires_at = None
    doctor.otp_attempts = 0
    db.commit()

    return {"detail": "Email verified successfully"}


def resend_email_otp(email: str, db: Session) -> dict:
    doctor = db.query(Doctor).filter(Doctor.email == email.lower().strip()).first()
    if not doctor:
        return {"detail": "If that email is registered, a new code has been sent.", "otp": None}

    if doctor.is_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    now = datetime.now(timezone.utc)
    if doctor.otp_last_sent_at:
        last_sent = doctor.otp_last_sent_at
        if last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=timezone.utc)
        elapsed = (now - last_sent).total_seconds()
        if elapsed < _OTP_RESEND_COOLDOWN_SECONDS:
            remaining = int(_OTP_RESEND_COOLDOWN_SECONDS - elapsed)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {remaining} seconds before requesting a new code.",
            )

    otp = _generate_otp()
    doctor.otp_code = _hash_otp(otp)
    doctor.otp_expires_at = now + timedelta(seconds=_OTP_TTL_SECONDS)
    doctor.otp_attempts = 0
    doctor.otp_last_sent_at = now
    db.commit()

    return {
        "detail": "A new verification code has been sent.",
        "otp": otp,
        "email": doctor.email,
        "full_name": doctor.full_name,
    }


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
