import hashlib
import hmac
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from auth import create_access_token, hash_password, verify_password
from enums import UserRole
from mail import send_otp_email, send_reset_email
from models import User

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

ENC_SECRET = os.getenv("ENC_SECRET_KEY")
TOKEN_TTL_SECONDS = 30 * 60  
OTP_TTL_SECONDS = 10 * 60  
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 60

PASSWORD_RULES = (
    "Password must be at least 8 characters and contain: "
    "one uppercase letter, one lowercase letter, one digit, "
    "and one special character (!@#$%^&*()_+-=[]{}|;:',.<>?/`~)."
)

def make_reset_token(email: str, timestamp: int) -> str:
    """Build a signed password reset token."""
    message = f"{email.lower().strip()}:{timestamp}".encode()
    return hmac.new(ENC_SECRET.encode(), message, hashlib.sha256).hexdigest()


def generate_reset_token(email: str) -> tuple[str, int]:
    """Generate a password reset token and timestamp."""
    ts = int(time.time())
    return make_reset_token(email, ts), ts


def verify_reset_token(email: str, token: str, timestamp: int) -> bool:
    """Check whether a reset token is still valid."""
    if int(time.time()) - timestamp > TOKEN_TTL_SECONDS :
        return False
    expected = make_reset_token(email, timestamp)
    return hmac.compare_digest(expected, token)


def generate_otp() -> str:
    """Generate a six-digit OTP code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    """Hash an OTP code for storage."""
    return hashlib.sha256(otp.encode()).hexdigest()


def validate_password(password: str) -> None:
    """Validate password strength rules."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail=PASSWORD_RULES)
    if not re.search(r"[A-Z]", password):
        raise HTTPException(status_code=400, detail=PASSWORD_RULES)
    if not re.search(r"[a-z]", password):
        raise HTTPException(status_code=400, detail=PASSWORD_RULES)
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail=PASSWORD_RULES)
    if not re.search(r"[!@#$%^&*()\-_+=\[\]{}|;:'\",.<>?/`~\\]", password):
        raise HTTPException(status_code=400, detail=PASSWORD_RULES)


async def register_user(
    email: str,
    password: str,
    full_name: str,
    db: Session,
    role: UserRole = UserRole.CLINICIAN,
) -> dict:
    """Register a new user and send an email verification OTP."""
    validate_password(password)

    existing = (
        db.query(User).filter(User.email == email.lower().strip()).first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    otp = generate_otp()
    now = datetime.now(timezone.utc)

    user = User(
        email=email.lower().strip(),
        hashed_password=hash_password(password),
        full_name=full_name.strip(),
        role=role.value,
        is_verified=False,
        otp_code=hash_otp(otp),
        otp_expires_at=now + timedelta(seconds=OTP_TTL_SECONDS),
        otp_attempts=0,
        otp_last_sent_at=now,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    await send_otp_email(user.email, otp, user.full_name)

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "is_verified": False,
    }


async def login_user(email: str, password: str, db: Session) -> dict:
    """Authenticate a user and refresh OTP if unverified."""
    user = (
        db.query(User).filter(User.email == email.lower().strip()).first()
    )
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id)

    if not user.is_verified:
        otp = generate_otp()
        now = datetime.now(timezone.utc)
        user.otp_code = hash_otp(otp)
        user.otp_expires_at = now + timedelta(seconds=OTP_TTL_SECONDS)
        user.otp_attempts = 0
        user.otp_last_sent_at = now
        db.commit()
        await send_otp_email(user.email, otp, user.full_name)

    return {
        "access_token": token,
        "token_type": "bearer",
        "is_verified": user.is_verified,
    }


def verify_email_otp(email: str, otp: str, db: Session) -> dict:
    """Verify an email OTP code for a user."""
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email")

    if user.is_verified:
        return {"detail": "Email already verified"}

    if user.otp_attempts >= OTP_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please request a new code.",
        )

    if not user.otp_code or not user.otp_expires_at:
        raise HTTPException(
            status_code=400,
            detail="No verification code found. Please request a new code.",
        )

    now = datetime.now(timezone.utc)
    expires = user.otp_expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if now > expires:
        raise HTTPException(
            status_code=400,
            detail="Verification code has expired. Please request a new code.",
        )

    if not hmac.compare_digest(hash_otp(otp), user.otp_code):
        user.otp_attempts += 1
        db.commit()
        remaining = OTP_MAX_ATTEMPTS - user.otp_attempts
        raise HTTPException(
            status_code=400,
            detail=f"Incorrect code. {remaining} attempt{'s' if remaining != 1 else ''} remaining.",
        )

    user.is_verified = True
    user.otp_code = None
    user.otp_expires_at = None
    user.otp_attempts = 0
    db.commit()

    return {"detail": "Email verified successfully"}


async def resend_email_otp(email: str, db: Session) -> dict:
    """Send a fresh email OTP if resend rules allow it."""
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user:
        return {"detail": "If that email is registered, a new code has been sent."}

    if user.is_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    now = datetime.now(timezone.utc)
    if user.otp_last_sent_at:
        last_sent = user.otp_last_sent_at
        if last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=timezone.utc)
        elapsed = (now - last_sent).total_seconds()
        if elapsed < OTP_RESEND_COOLDOWN_SECONDS:
            remaining = int(OTP_RESEND_COOLDOWN_SECONDS - elapsed)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {remaining} seconds before requesting a new code.",
            )

    otp = generate_otp()
    user.otp_code = hash_otp(otp)
    user.otp_expires_at = now + timedelta(seconds=OTP_TTL_SECONDS)
    user.otp_attempts = 0
    user.otp_last_sent_at = now
    db.commit()

    await send_otp_email(user.email, otp, user.full_name)

    return {"detail": "A new verification code has been sent."}


async def forgot_password(email: str, db: Session) -> dict:
    """Send a password reset link if the account exists."""
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if user:
        token, timestamp = generate_reset_token(email)
        reset_link = (
            f"{FRONTEND_URL}/reset-password"
            f"?email={user.email}"
            f"&token={token}"
            f"&ts={timestamp}"
        )
        await send_reset_email(user.email, reset_link)

    return {"detail": "If that email is registered you will receive a reset link shortly."}


def reset_password(email: str, token: str, timestamp: int, new_password: str, db: Session) -> dict:
    """Reset a user's password using a valid reset link."""
    validate_password(new_password)

    if not verify_reset_token(email, token, timestamp):
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")

    user.hashed_password = hash_password(new_password)
    db.commit()
    return {"detail": "Password updated successfully"}
