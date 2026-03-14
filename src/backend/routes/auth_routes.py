"""Auth route handlers: register, login, me, forgot/reset password."""

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from auth import get_current_doctor
from database import get_db
from mail import send_reset_email
from models import Doctor
from schemas import (
    DoctorLoginRequest,
    DoctorRegisterRequest,
    DoctorResponse,
    ErrorResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    TokenResponse,
)
from services.auth_service import forgot_password, login_doctor, register_doctor, reset_password

router = APIRouter(tags=["Auth"])


@router.post(
    "/auth/register",
    response_model=TokenResponse,
    summary="Register a new doctor",
    description="Create a new doctor account and return a JWT access token. "
    "Password must be at least 6 characters.",
    responses={
        200: {"description": "Registration successful, JWT token returned"},
        400: {
            "model": ErrorResponse,
            "description": "Validation error (short password or email already registered)",
        },
        422: {"description": "Request body validation error"},
    },
)
def register(body: DoctorRegisterRequest, db: Session = Depends(get_db)):
    return register_doctor(body.email, body.password, body.full_name, db)


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
def login(body: DoctorLoginRequest, db: Session = Depends(get_db)):
    return login_doctor(body.email, body.password, db)


@router.post(
    "/auth/forgot-password",
    summary="Request a password reset email",
    responses={200: {"description": "Reset email sent if account exists"}},
)
async def forgot_password_route(
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
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
        background_tasks.add_task(send_reset_email, result["email"], reset_link)

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
