"""
GridPulse AI — GenAI Grid Copilot Engine  (services/copilot_engine.py)

Purpose
-------
Thin, well-typed wrapper around the Google Gemini API that transforms a
structured grid-health context block + an operator's natural-language query
into a precise, analytically-grounded response.

Architecture
------------
                ┌─────────────────┐
  operator ──▶  │  CopilotRequest │
  query         └────────┬────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   ContextRetriever     │  (live DB queries → factual snapshot)
            └────────────┬───────────┘
                         │  context_block (plain text)
                         ▼
            ┌────────────────────────┐
            │   GridCopilot          │
            │   _build_messages()    │  (system prompt + context + query)
            │   Gemini Flash 2.0     │
            └────────────┬───────────┘
                         │  answer_text
                         ▼
                ┌────────────────┐
                │ CopilotResponse│  → FastAPI → operator
                └────────────────┘

System Prompt Design
--------------------
The system prompt enforces four behavioural constraints on the model:
  1. Identity — "You are GridPulse Copilot, an Expert Power Systems Grid Operator."
  2. Data grounding — respond only using facts in the GRID HEALTH SNAPSHOT.
  3. Format — cite specific meter IDs, INR figures, voltages, risk scores.
  4. Scope enforcement — gracefully refuse non-grid queries.

Design Decisions
----------------
• **google-genai 2.x SDK** — uses the modern `genai.Client` async interface.
• **Gemini 2.0 Flash** — optimal cost/latency for production copilot traffic.
  Easily swappable via GEMINI_MODEL env var.
• **Temperature 0.2** — factual Q&A benefits from near-deterministic output.
• **Max output 1024 tokens** — enough for a detailed grid report; avoids
  runaway costs on adversarial long-form prompts.
• **Graceful API error handling** — any Gemini API exception is caught and
  converted to a structured CopilotError instead of a 500 crash.
• **Strict context separation** — this module contains ZERO SQL.  All DB work
  lives in context_retriever.py.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

from config import get_settings

logger = logging.getLogger("gridpulse.copilot")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.0-flash"
TEMPERATURE   = 0.2
MAX_TOKENS    = 1024

# ── System prompt template ────────────────────────────────────────────────────
# Uses a single {context} placeholder filled at call time with the live DB snapshot.
# The prompt is designed to:
#   • Assert a strict expert identity and data-only policy.
#   • Instruct the model to always quote specific meter IDs / INR values.
#   • Provide a graceful fallback for out-of-scope questions.

_SYSTEM_PROMPT_TEMPLATE = """\
You are GridPulse Copilot — an Expert Power Systems Grid Operator and \
real-time grid analytics assistant deployed by a utility company managing \
smart meter infrastructure across India.

## Your Mission
Provide precise, actionable grid intelligence to field operators and \
engineers based exclusively on the live telemetry snapshot injected below. \
Every answer must reference specific data points: meter IDs, voltage/current \
readings, INR financial figures, anomaly types, or risk scores from the snapshot.

## Rules You Must Follow
1. **Use only the data below.** Never invent or extrapolate figures beyond \
what the snapshot shows.
2. **Always quote specifics.** If asked about losses, cite the exact INR \
amount and the meter ID it belongs to.
3. **Speak like an expert.** Use power-systems vocabulary: voltage sag, \
line tapping, power factor, outage risk, reactive load, etc.
4. **Stay in scope.** If the operator's question has no connection to \
electrical grid operations, financial losses, meter anomalies, or load \
forecasting, respond with exactly:
   "I'm GridPulse Copilot. I can only assist with grid operations, \
telemetry analysis, anomaly investigation, and financial impact queries. \
Please rephrase your question in that context."
5. **Be concise but complete.** Use bullet points, numeric tables, or \
short paragraphs as appropriate. Operators need answers in seconds, not \
essays.

## Live Grid Health Data (authoritative, do not contradict)
```
{context}
```
"""


# ── Response types ────────────────────────────────────────────────────────────

@dataclass
class CopilotResponse:
    """Successful copilot answer."""
    answer:     str
    model:      str
    context_chars: int    # size of injected context for observability
    input_tokens:  int | None = None
    output_tokens: int | None = None


@dataclass
class CopilotError:
    """Structured error returned when the LLM call fails."""
    error:   str
    detail:  str


# ── Engine ────────────────────────────────────────────────────────────────────

class GridCopilot:
    """
    Stateless LLM wrapper.  Instantiate once (singleton); call `ask_copilot`
    per operator request.

    Parameters
    ----------
    api_key : Gemini API key.  Defaults to GEMINI_API_KEY env var if omitted.
    model   : Gemini model name.  Defaults to GEMINI_MODEL env var or
              ``gemini-2.0-flash``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model:   str | None = None,
    ) -> None:
        cfg = get_settings()
        resolved_key = api_key or cfg.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is not set.  Add it to your .env file:\n"
                "  GEMINI_API_KEY=AIza..."
            )

        self._model  = model or cfg.GEMINI_MODEL or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = genai.Client(api_key=resolved_key)

        logger.info(
            "GridCopilot initialised — model=%s  temperature=%.1f  max_tokens=%d",
            self._model, TEMPERATURE, MAX_TOKENS,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def ask_copilot(
        self,
        operator_query: str,
        context_block:  str,
    ) -> CopilotResponse | CopilotError:
        """
        Send an operator query to Gemini, grounded by the live context block.

        Parameters
        ----------
        operator_query : Free-text question from the grid operator.
        context_block  : Plain-text grid health snapshot from ContextRetriever.

        Returns
        -------
        CopilotResponse on success, CopilotError on any API/network failure.
        """
        if not operator_query.strip():
            return CopilotError(
                error="EmptyQuery",
                detail="The operator query cannot be empty.",
            )

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context_block)
        context_chars = len(context_block)

        logger.info(
            "Copilot query received | model=%s  context_chars=%d  query_len=%d",
            self._model, context_chars, len(operator_query),
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=operator_query,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                    candidate_count=1,
                ),
            )

            answer = response.text or ""

            # Token usage (best-effort — may be None for some model versions)
            usage = getattr(response, "usage_metadata", None)
            in_tok  = getattr(usage, "prompt_token_count",     None)
            out_tok = getattr(usage, "candidates_token_count", None)

            logger.info(
                "Copilot response | in_tokens=%s  out_tokens=%s  answer_chars=%d",
                in_tok, out_tok, len(answer),
            )

            return CopilotResponse(
                answer=answer,
                model=self._model,
                context_chars=context_chars,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc, exc_info=True)
            return CopilotError(
                error="LLMCallFailed",
                detail=str(exc),
            )


# ── Module-level singleton ────────────────────────────────────────────────────

_copilot: GridCopilot | None = None


def get_copilot() -> GridCopilot:
    """
    Return the module-level GridCopilot singleton.

    Raises ``ValueError`` if GEMINI_API_KEY is not configured.
    Call this inside a FastAPI dependency or lifespan handler — not at import
    time — so that missing API keys produce a clean startup error, not an
    import crash.
    """
    global _copilot
    if _copilot is None:
        _copilot = GridCopilot()
    return _copilot
