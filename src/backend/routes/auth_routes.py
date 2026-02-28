"""Auth route handlers: register, login, me."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from auth import get_current_doctor
from database import get_db
from models import Doctor
from schemas import (
    DoctorLoginRequest,
    DoctorRegisterRequest,
    DoctorResponse,
    ErrorResponse,
    TokenResponse,
)
from services.auth_service import login_doctor, register_doctor

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
