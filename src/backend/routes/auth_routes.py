"""Auth route handlers: register, login, me, forgot/reset password, email OTP verification."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth import get_current_doctor
from database import get_db
from mail import send_otp_email, send_reset_email
from models import Doctor
from schemas import (
    DoctorLoginRequest,
    DoctorRegisterRequest,
    DoctorResponse,
    ErrorResponse,
    ForgotPasswordRequest,
    ResendOtpRequest,
    ResetPasswordRequest,
    TokenResponse,
    VerifyOtpRequest,
)
from services.auth_service import (
    forgot_password,
    login_doctor,
    register_doctor,
    resend_email_otp,
    reset_password,
    verify_email_otp,
)

router = APIRouter(tags=["Auth"])


@router.post(
    "/auth/register",
    response_model=TokenResponse,
    summary="Register a new doctor",
    description="Create a new doctor account and return a JWT access token. "
    "Password must be at least 6 characters. Account starts unverified.",
    responses={
        200: {"description": "Registration successful, JWT token returned"},
        400: {
            "model": ErrorResponse,
            "description": "Validation error (short password or email already registered)",
        },
        422: {"description": "Request body validation error"},
    },
)
async def register(
    body: DoctorRegisterRequest,
    db: Session = Depends(get_db),
):
    result = register_doctor(body.email, body.password, body.full_name, db)
    if result.get("otp"):
        await send_otp_email(result["email"], result["otp"], result["full_name"])
    return TokenResponse(
        access_token=result["access_token"],
        is_verified=result["is_verified"],
    )


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Login",
    description="Authenticate with email and password. Returns a JWT access token valid for 8 hours.",
    responses={
        200: {"description": "Login successful, JWT token returned"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid email or password",
        },
    },
)
async def login(
    body: DoctorLoginRequest,
    db: Session = Depends(get_db),
):
    result = login_doctor(body.email, body.password, db)
    if result.get("otp"):
        await send_otp_email(result["email"], result["otp"], result["full_name"])
    return TokenResponse(
        access_token=result["access_token"],
        is_verified=result["is_verified"],
    )


@router.post(
    "/auth/verify-email-otp",
    summary="Verify email with OTP code",
    responses={
        200: {"description": "Email verified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid or expired code"},
        429: {"model": ErrorResponse, "description": "Too many attempts"},
    },
)
def verify_otp_route(body: VerifyOtpRequest, db: Session = Depends(get_db)):
    return verify_email_otp(body.email, body.otp, db)


@router.post(
    "/auth/resend-email-otp",
    summary="Resend email verification OTP",
    responses={
        200: {"description": "New code sent"},
        429: {"model": ErrorResponse, "description": "Cooldown active"},
    },
)
async def resend_otp_route(
    body: ResendOtpRequest,
    db: Session = Depends(get_db),
):
    result = resend_email_otp(body.email, db)
    if result.get("otp"):
        await send_otp_email(result["email"], result["otp"], result["full_name"])
    return {"detail": result["detail"]}


@router.post(
    "/auth/forgot-password",
    summary="Request a password reset email",
    responses={200: {"description": "Reset email sent if account exists"}},
)
async def forgot_password_route(
    body: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    result = forgot_password(body.email, db)

    if result["token"] is not None:
        frontend_url = "http://localhost:5173"
        reset_link = (
            f"{frontend_url}/reset-password"
            f"?email={result['email']}"
            f"&token={result['token']}"
            f"&ts={result['timestamp']}"
        )
        await send_reset_email(result["email"], reset_link)

    # Always return the same message — never reveal if email exists
    return {"detail": "If that email is registered you will receive a reset link shortly."}


@router.post(
    "/auth/reset-password",
    summary="Reset password using token from email",
    responses={
        200: {"description": "Password updated"},
        400: {"model": ErrorResponse, "description": "Invalid/expired token or weak password"},
    },
)
def reset_password_route(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    return reset_password(body.email, body.token, body.timestamp, body.new_password, db)


@router.get(
    "/auth/me",
    response_model=DoctorResponse,
    summary="Get current doctor profile",
    description="Return the authenticated doctor's profile information.",
    responses={
        200: {"description": "Doctor profile returned"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
    },
)
def get_me(current_doctor: Doctor = Depends(get_current_doctor)):
    return current_doctor
