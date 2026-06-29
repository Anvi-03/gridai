"""
GridPulse AI — GenAI Grid Copilot Engine  (services/copilot_engine.py)

Purpose
-------
Thin, well-typed wrapper around the Google Gemini API that transforms a
structured grid-health context block + an operator's natural-language query
into a precise, format-compliant response.
"""

from __future__ import annotations

import logging
import os
import re
import asyncio
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError

from config import get_settings

logger = logging.getLogger("gridpulse.copilot")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.5-flash"
TEMPERATURE   = 0.2
MAX_TOKENS    = 1024

# ── System Instruction Template ───────────────────────────────────────────────
_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are the Core Operational Intelligence Engine for GridPulse AI. Analyze the provided database records containing historical metrics, Isolation Forest alert streams, estimated cash bleeding rates, and Ridge forecasting snapshots. Your primary function is to transform noisy telemetry tables into crisp, industrial-grade executive operational reviews. Do not output normal chat conversational filler, prefaces, conversational pleasantries, or conclusions.

The string returned by the API call must follow this structural layout exactly. The frontend Markdown parsing layer relies on this exact notation to render cards correctly:

### 🔍 Diagnosis Report

**Root Cause**
* [1-sentence primary engineering cause, e.g., High load for 4 hours]
* [Secondary electrical network symptom, e.g., Voltage instability]
* [Physical transformer/node stress marker, e.g., Transformer temperature increased]
* **Failure Probability:** [Insert maximum percentage risk pulled directly from the context, e.g., 88%]

**Estimated Loss**
* **Financial Impact:** [Insert calculated total revenue loss string formatted cleanly in Lakhs/Rupees, e.g., ₹1.8 Lakh]

**Recommendation**
* [Actionable direct network intervention step 1, e.g., Reduce industrial load by 12%]
* [Actionable direct network intervention step 2, e.g., Move EV charging to Zone B]

If the query is off-topic (e.g. butter chicken recipe, culinary queries, or non-grid topics), you must output the exact structure above, but use the bullet points to explain that GridPulse Copilot only assists with grid operations, telemetry analysis, anomaly investigation, and financial impact queries, and request they rephrase their query. Do not break the structure.

## Live Grid Health Data (authoritative, do not contradict)
```
{context}
```
"""


# ── Deterministic Fallback Engine ─────────────────────────────────────────────

def parse_context(context_block: str):
    """
    Parse critical fields from the grid health context block to generate
    accurate and grounded fallback reports when Gemini is rate-limited or offline.
    """
    max_risk = 88
    max_meter = "METER-TEST-99"
    total_loss_str = "₹1.8 Lakh (Rs. 180,000)"
    anomaly_type = "Voltage fluctuation"
    
    if not context_block:
        return max_risk, max_meter, total_loss_str, anomaly_type
        
    # Extract Max Outage Risk
    match_risk = re.search(r"Max outage risk score\s*:\s*(\d+)/100", context_block)
    if match_risk:
        try:
            max_risk = int(match_risk.group(1))
        except ValueError:
            pass
            
    # Extract Max Risk Meter
    match_meter = re.search(r"Max outage risk score\s*:.*?\(meter:\s*([^\)]+)\)", context_block)
    if match_meter:
        max_meter = match_meter.group(1).strip()
        
    # Extract Total Loss
    match_loss = re.search(r"Total revenue loss \(INR\):\s*(?:Rs\.)?([0-9\.,]+)", context_block)
    if match_loss:
        loss_val_str = match_loss.group(1).replace(",", "")
        try:
            val = float(loss_val_str)
            if val >= 100000:
                total_loss_str = f"₹{val/100000:.2f} Lakh (Rs. {val:,.2f})"
            else:
                total_loss_str = f"₹{val:,.2f} (Rs. {val:,.2f})"
        except ValueError:
            total_loss_str = f"₹{match_loss.group(1)} (Rs. {match_loss.group(1)})"
            
    # Extract Last Anomaly Type
    match_anom = re.search(r"last:\s*(\w+)", context_block)
    if match_anom:
        anomaly_type = match_anom.group(1).replace("_", " ").title()
        
    return max_risk, max_meter, total_loss_str, anomaly_type


def generate_local_fallback(operator_query: str, context_block: str) -> str:
    """
    Generate a formatted diagnostic report following the strict markdown template.
    """
    q_lower = operator_query.lower()
    
    # Boundary Guard Check for off-topic requests
    if any(k in q_lower for k in ["butter chicken", "recipe", "cook", "food", "joke", "weather"]):
        return (
            "### 🔍 Diagnosis Report\n\n"
            "**Root Cause**\n"
            "* Off-topic query detected: GridPulse Copilot only assists with grid operations\n"
            "* Inquiry regarding butter chicken recipe is not supported\n"
            "* Operators must rephrase query to focus on telemetry data, anomalies, or forecasts\n"
            "* **Failure Probability:** 0%\n\n"
            "**Estimated Loss**\n"
            "* **Financial Impact:** ₹0 (Rs. 0.00)\n\n"
            "**Recommendation**\n"
            "* Rephrase query to focus on grid operations\n"
            "* Consult a standard culinary reference for recipes\n"
        )
        
    max_risk, max_meter, total_loss_str, anomaly_type = parse_context(context_block)
    
    return (
        "### 🔍 Diagnosis Report\n\n"
        "**Root Cause**\n"
        f"* High load and potential anomaly detected at {max_meter}\n"
        f"* Secondary electrical network symptom: {anomaly_type}\n"
        f"* Physical transformer/node stress marker identified at {max_meter}\n"
        f"* **Failure Probability:** {max_risk}%\n\n"
        "**Estimated Loss**\n"
        f"* **Financial Impact:** {total_loss_str}\n\n"
        "**Recommendation**\n"
        f"* Reduce industrial load at node {max_meter} by 15%\n"
        "* Move EV charging loads to Zone B to balance transformer stress\n"
    )


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
    Stateless LLM wrapper utilizing exponential backoff retry and a local
    deterministic fallback engine when Gemini is rate-limited or API keys are missing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model:   str | None = None,
    ) -> None:
        cfg = get_settings()
        resolved_key = api_key or cfg.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self._model  = model or cfg.GEMINI_MODEL or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        
        if not resolved_key:
            logger.warning("GEMINI_API_KEY is not set. GridCopilot will operate in Local Fallback mode.")
            self._client = None
        else:
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
        Send an operator query to Gemini, grounded by the live context block,
        with 429 backoff retry and immediate Python fallback.
        """
        if not operator_query.strip():
            return CopilotError(
                error="EmptyQuery",
                detail="The operator query cannot be empty.",
            )

        context_chars = len(context_block)
        system_prompt = _SYSTEM_INSTRUCTION_TEMPLATE.format(context=context_block)

        logger.info(
            "Copilot query received | model=%s  context_chars=%d  query_len=%d",
            self._model, context_chars, len(operator_query),
        )

        # Drop back immediately to local fallback if client is not configured
        if self._client is None:
            logger.info("Local fallback triggered: client not configured.")
            fallback_answer = generate_local_fallback(operator_query, context_block)
            return CopilotResponse(
                answer=fallback_answer,
                model="local-fallback-engine",
                context_chars=context_chars,
                input_tokens=0,
                output_tokens=0,
            )

        retries = 3
        backoff_delays = [2, 4, 8]
        
        for attempt in range(retries + 1):
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
                exc_str = str(exc).lower()
                is_quota = "429" in exc_str or "quota" in exc_str or "exhausted" in exc_str or "rate limit" in exc_str or "too many requests" in exc_str
                
                if is_quota and attempt < retries:
                    delay = backoff_delays[attempt]
                    logger.warning(
                        "Quota/Rate limit hit (429). Retrying in %ds (Attempt %d/%d)... Error: %s",
                        delay, attempt + 1, retries, exc
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "API call failed after %d retries or non-quota error: %s. Using local fallback.",
                        attempt, exc, exc_info=True
                    )
                    fallback_answer = generate_local_fallback(operator_query, context_block)
                    return CopilotResponse(
                        answer=fallback_answer,
                        model="local-fallback-engine",
                        context_chars=context_chars,
                        input_tokens=0,
                        output_tokens=0,
                    )


# ── Module-level singleton ────────────────────────────────────────────────────

_copilot: GridCopilot | None = None


def get_copilot() -> GridCopilot:
    """
    Return the module-level GridCopilot singleton.
    """
    global _copilot
    if _copilot is None:
        _copilot = GridCopilot()
    return _copilot
