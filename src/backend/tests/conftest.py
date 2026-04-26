
import os

os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-pytest-only!!"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

import database as _db_module
from database import Base, get_db

_TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_TEST_ENGINE)

_db_module.engine = _TEST_ENGINE
_db_module.SessionLocal = _TestSession

from auth import create_access_token, hash_password 
from enums import UserRole  
from models import User 
from tests.helpers import make_dataset_zip, make_torchscript_bytes  


# ── Database lifecycle ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _tables():
    """Create all tables before each test, drop them after."""
    Base.metadata.create_all(bind=_TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=_TEST_ENGINE)


@pytest.fixture
def db(_tables):
    session = _TestSession()
    try:
        yield session
    finally:
        session.close()


# ── User fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def developer(db):
    user = User(
        email="dev@test.com",
        hashed_password=hash_password("password123"),
        full_name="Test Developer",
        role=UserRole.DEVELOPER.value,
        is_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def other_developer(db):
    user = User(
        email="other@test.com",
        hashed_password=hash_password("password123"),
        full_name="Other Developer",
        role=UserRole.DEVELOPER.value,
        is_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def dev_token(developer):
    return create_access_token(developer.id)


@pytest.fixture
def other_dev_token(other_developer):
    return create_access_token(other_developer.id)


# ── File fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def model_bytes():
    return make_torchscript_bytes()


@pytest.fixture
def dataset_bytes():
    return make_dataset_zip()


# ── Azure mock (autouse — no test ever touches real Azure) ────────────────

@pytest.fixture(autouse=True)
def mock_azure():
    """Patch upload/download/delete at the service-module level (where names are bound)."""
    with (
        patch("services.calibration_service.upload_to_bucket"),
        patch("services.calibration_service.download_from_bucket", return_value=b"{}"),
        patch("services.calibration_service.delete_from_bucket"),
        patch("services.published_model_service.upload_to_bucket"),
        patch("services.published_model_service.download_from_bucket", return_value=b"{}"),
    ):
        yield


# ── HTTP test client ──────────────────────────────────────────────────────

@pytest.fixture
def client(db):
    """TestClient with the in-memory test DB injected via dependency override."""
    from api import app
    from fastapi.testclient import TestClient

    def _override():
        yield db

    app.dependency_overrides[get_db] = _override
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()
