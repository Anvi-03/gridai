"""
GridPulse AI — Shared FastAPI Dependencies  (api/v1/deps.py)

Provides reusable dependency callables that can be injected into any
route via FastAPI's ``Depends()`` mechanism.

Current dependencies
--------------------
get_current_user
    Extracts the JWT bearer token from the Authorization header, decodes
    and verifies it, and returns the authenticated username string.

    Any route decorated with ``Depends(get_current_user)`` will:
      • Accept requests carrying a valid, unexpired JWT → continue normally.
      • Reject all other requests with HTTP 401 Unauthorized.

Usage
-----
    from api.v1.deps import get_current_user

    @router.post("/protected-endpoint")
    async def my_endpoint(
        body: MyRequest,
        current_user: str = Depends(get_current_user),
    ) -> MyResponse:
        # current_user is the authenticated username string
        ...
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from core.security import decode_access_token

# ── OAuth2 scheme ──────────────────────────────────────────────────────────────
# tokenUrl tells FastAPI's /docs UI where the lock icon should POST credentials.
# It must match the exact path of the /login endpoint (no leading slash, no host).

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")


# ── Dependency ─────────────────────────────────────────────────────────────────

async def get_current_user(
    token: Annotated[str, Depends(_oauth2_scheme)],
) -> str:
    """
    Decode the incoming JWT and return the authenticated username.

    FastAPI will automatically extract the token from the
    ``Authorization: Bearer <token>`` header via the OAuth2PasswordBearer
    scheme.  If the header is absent, malformed, or the token is
    invalid/expired, a 401 Unauthorized response is raised before this
    function body is ever executed.

    Returns
    -------
    str
        The ``sub`` claim from the JWT payload (the username).

    Raises
    ------
    HTTPException(401)
        If the token is missing, expired, or has an invalid signature.
    HTTPException(401)
        If the token payload is missing the ``sub`` claim.
    """
    payload = decode_access_token(token)

    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing the 'sub' claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username
