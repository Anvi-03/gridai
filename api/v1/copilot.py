"""
GridPulse AI — Copilot API Router  (api/v1/copilot.py)

Exposes:
    POST /api/v1/copilot/query
        Accepts a natural-language operator question, injects live grid
        telemetry context, and returns a Gemini-powered analytical response.

    GET  /api/v1/copilot/context
        Returns the raw context block that would be injected on the next
        query — useful for debugging and frontend pre-loading.

    GET  /api/v1/copilot/health
        Verifies the copilot engine is configured and reachable.

Design Decisions
----------------
• **APIRouter** — mounted under /api/v1 in main.py via include_router().
  Keeps copilot logic fully decoupled from the telemetry ingest routes.
• **Pydantic I/O models** — CopilotQueryRequest / CopilotQueryResponse are
  defined here so the OpenAPI schema is clean and self-documenting.
• **Async dependency injection** — the DB session is injected by FastAPI's
  DI system; the copilot singleton is retrieved via get_copilot().
• **CopilotError → HTTP 502** — LLM API failures produce a structured 502
  (Bad Gateway) rather than a 500, signalling to the client that the backend
  service itself is healthy but the upstream LLM is not.
• **Context endpoint** — exposing /copilot/context lets frontend devs
  inspect exactly what the LLM sees without burning tokens.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.v1.deps import get_current_user
from database import get_db
from services.context_retriever import get_context_retriever
from services.copilot_engine import CopilotError, get_copilot
from config import settings

logger = logging.getLogger("gridpulse.copilot.router")

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/copilot",
    tags=["GenAI Copilot"],
)


# ── Pydantic I/O models ───────────────────────────────────────────────────────

class CopilotQueryRequest(BaseModel):
    """Request body for the copilot query endpoint."""

    message: str = Field(
        min_length=3,
        max_length=2000,
        description="Natural-language question from the grid operator.",
        examples=[
            "Which meter had the highest revenue loss today?",
            "Why did consumption spike in the last hour?",
            "List all meters with outage risk above 70.",
        ],
    )

    model_config = {"str_strip_whitespace": True}


class CopilotQueryResponse(BaseModel):
    """Response envelope for a successful copilot query."""

    answer:        str   = Field(description="Gemini-generated analytical response.")
    model:         str   = Field(description="LLM model used.")
    context_chars: int   = Field(description="Size of injected context (chars).")
    input_tokens:  int | None = Field(default=None, description="Prompt token count.")
    output_tokens: int | None = Field(default=None, description="Response token count.")


class CopilotContextResponse(BaseModel):
    """Raw context block returned by the /context debug endpoint."""

    context: str = Field(description="Plain-text grid health snapshot.")
    chars:   int = Field(description="Character count of the context block.")


class CopilotHealthResponse(BaseModel):
    """Health check response for the copilot subsystem."""

    status:  str = Field(description="'ready' or 'misconfigured'.")
    model:   str = Field(description="LLM model that will be used.")
    detail:  str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=CopilotQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask the Grid Copilot a question",
    description=(
        "Accepts a natural-language question from a grid operator. "
        "Automatically injects a live telemetry snapshot (last 24 h) into "
        "the LLM prompt so answers are grounded in real data — never guessed."
    ),
    responses={
        400: {"description": "Empty or too-short query."},
        502: {"description": "Upstream LLM API call failed."},
        503: {"description": "Copilot engine not configured (missing API key)."},
    },
)
async def query_copilot(
    body: CopilotQueryRequest,
    db:   Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[str, Depends(get_current_user)],
) -> CopilotQueryResponse:
    """
    Orchestrate:
    1. Retrieve live grid context from PostgreSQL.
    2. Inject context + operator query into the Gemini system prompt.
    3. Return the model's answer.
    """

    # ── 1. Resolve copilot engine (raises 503 if key missing) ─────────────────
    try:
        copilot = get_copilot()
    except ValueError as exc:
        logger.error("Copilot engine not configured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GridCopilot is not configured. "
                "Set GEMINI_API_KEY in your .env file."
            ),
        )

    # ── 2. Build live context block ────────────────────────────────────────────
    retriever     = get_context_retriever()
    context_block = await retriever.build_context(db)

    logger.info(
        "Copilot context built | chars=%d  query=%r",
        len(context_block),
        body.message[:80],
    )

    # ── 3. Call Gemini ─────────────────────────────────────────────────────────
    result = await copilot.ask_copilot(
        operator_query=body.message,
        context_block=context_block,
    )

    if isinstance(result, CopilotError):
        logger.error("LLM call failed: %s — %s", result.error, result.detail)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM call failed ({result.error}): {result.detail}",
        )

    return CopilotQueryResponse(
        answer=result.answer,
        model=result.model,
        context_chars=result.context_chars,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


@router.get(
    "/context",
    response_model=CopilotContextResponse,
    summary="Inspect the live grid context block",
    description=(
        "Returns the exact plain-text snapshot that would be injected into the "
        "LLM system prompt on the next /query call. Use this to debug context "
        "quality or pre-load data on the frontend without burning tokens."
    ),
)
async def get_grid_context(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CopilotContextResponse:
    """Return the live grid health snapshot without invoking the LLM."""
    retriever     = get_context_retriever()
    context_block = await retriever.build_context(db)
    return CopilotContextResponse(context=context_block, chars=len(context_block))


@router.get(
    "/health",
    response_model=CopilotHealthResponse,
    summary="Copilot subsystem health check",
)
async def copilot_health() -> CopilotHealthResponse:
    """
    Verifies that the copilot engine is configured with a valid API key.
    Does NOT make a live Gemini call — just checks configuration.
    """
    api_key = settings.GEMINI_API_KEY
    model   = settings.GEMINI_MODEL

    if not api_key:
        return CopilotHealthResponse(
            status="misconfigured",
            model=model,
            detail="GEMINI_API_KEY is not set in environment.",
        )

    return CopilotHealthResponse(
        status="ready",
        model=model,
        detail="Copilot engine configured and ready.",
    )
