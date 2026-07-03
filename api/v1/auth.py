"""
GridPulse AI — Authentication Router  (api/v1/auth.py)

Exposes:
    POST /api/v1/auth/register
        Create a new operator account.  Password is bcrypt-hashed before
        storage; the plain-text credential is never persisted.

    POST /api/v1/auth/login
        OAuth2-compliant token endpoint.  Accepts form-encoded credentials
        (username + password via OAuth2PasswordRequestForm), validates
        against the database, and returns a signed JWT access token.

Design notes
------------
• OAuth2PasswordRequestForm is used for /login so the endpoint is
  compatible with FastAPI's /docs interactive UI (the lock icon) and any
  standard OAuth2 client library out of the box.

• Username lookup uses an exact case-sensitive match.  For case-insensitive
  usernames, add `func.lower(User.username) == username.lower()` to the
  WHERE clause and a functional index in the migration.

• The 401 error for wrong credentials deliberately does NOT distinguish
  "user not found" from "wrong password" to prevent username enumeration.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import create_access_token, get_password_hash, verify_password
from database import get_db
from schemas import User

logger = logging.getLogger("gridpulse.auth")

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ── Pydantic I/O models ───────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Request body for account registration."""

    username: str = Field(
        min_length=3,
        max_length=64,
        description="Unique operator username (3–64 characters).",
        examples=["grid_operator_01"],
    )
    password: str = Field(
        min_length=8,
        max_length=128,
        description="Plain-text password (min 8 characters; hashed before storage).",
    )

    model_config = {"str_strip_whitespace": True}

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        """Reject usernames with characters that could cause SQL/display issues."""
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
        if not all(c in allowed for c in v):
            raise ValueError(
                "Username may only contain letters, digits, underscores, hyphens, and dots."
            )
        return v


class RegisterResponse(BaseModel):
    """Confirmation payload returned after successful registration."""
    message:  str = Field(description="Human-readable confirmation.")
    username: str = Field(description="The registered username.")


class TokenResponse(BaseModel):
    """OAuth2 token response payload."""
    access_token: str  = Field(description="Signed JWT bearer token.")
    token_type:   str  = Field(default="bearer", description="Always 'bearer'.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new operator account",
    responses={
        409: {"description": "Username already taken."},
        422: {"description": "Validation error (username/password constraints)."},
    },
)
async def register(
    body: RegisterRequest,
    db:   Annotated[AsyncSession, Depends(get_db)],
) -> RegisterResponse:
    """
    Hash the submitted password and persist a new User row.

    Returns 409 Conflict if the username is already registered so the
    frontend can prompt the user to choose a different name.
    """
    hashed = get_password_hash(body.password)
    user   = User(username=body.username, hashed_password=hashed)

    try:
        db.add(user)
        await db.flush()  # surface unique-constraint violations before commit
    except IntegrityError:
        await db.rollback()
        logger.warning("Registration failed — duplicate username: %r", body.username)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken. Please choose another.",
        )

    logger.info("New operator registered: %r", body.username)
    return RegisterResponse(
        message=f"Account '{body.username}' created successfully. You can now log in.",
        username=body.username,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Obtain a JWT access token",
    description=(
        "Accepts standard OAuth2 form-encoded credentials "
        "(``application/x-www-form-urlencoded``) and returns a signed JWT. "
        "The token must be included as ``Authorization: Bearer <token>`` on "
        "all protected endpoints."
    ),
    responses={
        401: {"description": "Invalid username or password."},
    },
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db:        Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Validate credentials against the database and issue a signed JWT.

    Uses a timing-safe bcrypt comparison so that wrong-password and
    no-such-user paths take the same wall-clock time (prevents oracle attacks).
    """
    _invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid username or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # ── 1. Fetch the user record ───────────────────────────────────────────────
    result = await db.execute(
        select(User).where(User.username == form_data.username)
    )
    user: User | None = result.scalar_one_or_none()

    # ── 2. Validate password (always call verify so timing is consistent) ─────
    if user is None or not verify_password(form_data.password, user.hashed_password):
        logger.warning("Failed login attempt for username: %r", form_data.username)
        raise _invalid

    # ── 3. Issue JWT ───────────────────────────────────────────────────────────
    token = create_access_token(data={"sub": user.username})
    logger.info("Operator %r logged in successfully.", user.username)

    return TokenResponse(access_token=token, token_type="bearer")
