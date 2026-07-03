"""
GridPulse AI — Core Security Utilities  (core/security.py)

Provides password hashing/verification via bcrypt and stateless JWT
token generation/decoding via PyJWT (HS256).

All functions are synchronous pure-Python — safe to call from both
sync and async contexts without wrapping in run_in_executor.

Usage
-----
    from core.security import (
        get_password_hash,
        verify_password,
        create_access_token,
        decode_access_token,
    )

Token lifecycle
---------------
    • create_access_token({"sub": username})   → signed JWT string
    • decode_access_token(token)               → {"sub": username, "exp": ...}
      Raises fastapi.HTTPException(401) on expiry or invalid signature.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext

from config import settings

# ── bcrypt context ─────────────────────────────────────────────────────────────
# schemes=["bcrypt"] uses bcrypt as the single active scheme.
# deprecated="auto" marks old schemes as deprecated during verification so
# passwords can be auto-upgraded on next login without breaking anything.

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password helpers ───────────────────────────────────────────────────────────

def get_password_hash(password: str) -> str:
    """Return the bcrypt hash of *password*.

    Always generates a fresh salt — two calls with the same password produce
    different hashes (which verify_password handles transparently).
    """
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if *plain_password* matches *hashed_password*.

    Uses constant-time comparison internally — safe against timing attacks.
    """
    return _pwd_context.verify(plain_password, hashed_password)


# ── JWT helpers ────────────────────────────────────────────────────────────────

def create_access_token(data: dict) -> str:
    """Sign *data* as a JWT access token with a UTC expiry stamp.

    A copy of *data* is made so the caller's dict is never mutated.
    The ``exp`` claim is injected automatically based on
    ``settings.JWT_EXPIRE_MINUTES``.

    Returns the compact (Base64URL-encoded) JWT string.
    """
    payload = data.copy()
    expire  = datetime.now(tz=timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    payload["exp"] = expire

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    """Decode and verify *token*; return the raw payload dict.

    Raises
    ------
    HTTPException(401)
        • If the token's signature is invalid (tampered).
        • If the token has expired (``exp`` in the past).
        • If the token is structurally malformed.

    On success the returned dict contains at least ``{"sub": <username>, "exp": <timestamp>}``.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Token is invalid or has expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise credentials_exception
