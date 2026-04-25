from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import User
from schemas import (
    LoginRequest,
    RegisterRequest,
    UserResponse,
    ErrorResponse,
    ForgotPasswordRequest,
    ResendOtpRequest,
    ResetPasswordRequest,
    TokenResponse,
    VerifyOtpRequest,
)
from services.auth_service import (
    forgot_password,
    login_user,
    register_user,
    resend_email_otp,
    reset_password,
    verify_email_otp,
)

router = APIRouter(tags=["Auth"])


@router.post(
    "/auth/register",
    response_model=TokenResponse,
    summary="Register a new user",
    description="Create a new user account and return a JWT access token. "
    "Password must be at least 8 characters with uppercase, lowercase, digit, and special character. "
    "Account starts unverified.",
    responses={
        200: {"description": "Registration successful, JWT token returned"},
        400: {
            "model": ErrorResponse,
            "description": "Validation error (weak password or email already registered)",
        },
        422: {"description": "Request body validation error"},
    },
)
async def register(
    body: RegisterRequest,
    db: Session = Depends(get_db),
):
    result = await register_user(body.email, body.password, body.full_name, db)
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
    body: LoginRequest,
    db: Session = Depends(get_db),
):
    result = await login_user(body.email, body.password, db)
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
    result = await resend_email_otp(body.email, db)
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
    return await forgot_password(body.email, db)


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
    response_model=UserResponse,
    summary="Get current user profile",
    description="Return the authenticated user's profile information.",
    responses={
        200: {"description": "User profile returned"},
        401: {
            "model": ErrorResponse,
            "description": "Invalid or expired token",
        },
    },
)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
