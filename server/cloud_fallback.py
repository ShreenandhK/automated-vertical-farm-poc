"""
cloud_fallback.py — Dual-API Cloud Vision Fallback (Anthropic Claude + Google Gemini)

When the local TFLite model's confidence falls below CONFIDENCE_THRESHOLD,
this module takes over.  It checks the .env file for whichever API key is
present — ANTHROPIC_API_KEY or GEMINI_API_KEY — and routes the image to
the corresponding cloud vision model.

Why support two providers?
──────────────────────────
Think of it like having two specialist doctors on call: if one is
unavailable (no API key configured), the system seamlessly consults the
other.  This gives the deployer flexibility to choose based on cost,
availability, or regional preference — without touching a single line of
application code.

Both providers return the exact same JSON shape so that main.py never
needs to care which cloud model answered.

Why async?
──────────
The Anthropic Python SDK provides an AsyncAnthropic client that plays
nicely with FastAPI's event loop.  For Gemini, the synchronous SDK call
is offloaded to a thread via asyncio.to_thread() so it doesn't block the
event loop either.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Environment & logging
# ──────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

# ──────────────────────────────────────────────
# Lazy client initialisation
# ──────────────────────────────────────────────
# Clients are created on first use, not at import time.  This avoids
# import-time crashes when an API key is missing (e.g. during tests or
# when only one provider is configured).
_anthropic_client: Optional[object] = None  # Will be anthropic.AsyncAnthropic
_gemini_client: Optional[object] = None      # Will be google.genai.Client


def _get_anthropic_client():
    """Return (and cache) the async Anthropic client.

    Raises ValueError if the API key is not set.
    """
    global _anthropic_client  # noqa: PLW0603

    if _anthropic_client is not None:
        return _anthropic_client

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env.")

    # Import here so the module loads even if `anthropic` isn't installed.
    import anthropic

    _anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("AsyncAnthropic client initialised for cloud fallback.")
    return _anthropic_client


def _get_gemini_client():
    """Return (and cache) a google-genai Client instance.

    The new google-genai SDK uses an explicit Client object rather than a
    module-level configure() call.  This mirrors how the Anthropic client
    works — one client per process, created lazily on first use.

    Raises ValueError if the API key is not set.
    """
    global _gemini_client  # noqa: PLW0603

    if _gemini_client is not None:
        return _gemini_client

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env.")

    # Import here so the module loads even if `google-genai` isn't installed.
    from google import genai

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Google Gemini client initialised for cloud fallback.")
    return _gemini_client


# ──────────────────────────────────────────────
# Shared prompt text
# ──────────────────────────────────────────────
# Both providers receive the same analysis instructions.  Keeping the
# prompt in one place ensures diagnostic consistency regardless of which
# cloud model answers.

_ANALYSIS_PROMPT: str = (
    "You are an expert agricultural pathologist specialising in plant "
    "disease identification for vertical-farm crops.\n\n"
    "Analyse the provided plant photograph and respond with ONLY a JSON "
    "object — no markdown fences, no commentary.  Use this exact schema:\n\n"
    '{\n'
    '  "disease_detected": true or false,\n'
    '  "disease_name": "<disease name or \'healthy\'>",\n'
    '  "confidence": "<high, medium, or low>",\n'
    '  "recommended_action": "<one-sentence plain-English recommendation>"\n'
    '}\n\n'
    "If the plant appears healthy, set disease_detected to false and "
    "disease_name to \"healthy\".\n"
    "If you cannot determine the condition, set disease_detected to false, "
    "disease_name to \"unknown\", and confidence to \"low\"."
)


# ──────────────────────────────────────────────
# Provider-specific analysis functions
# ──────────────────────────────────────────────

async def _analyze_with_anthropic(image_bytes: bytes, media_type: str) -> dict:
    """Send the image to Claude Vision (Anthropic) and return structured JSON.

    Uses the AsyncAnthropic client so the call is natively non-blocking.
    The image is base64-encoded in memory — no temp file needed.
    """
    client = _get_anthropic_client()

    image_b64: str = base64.b64encode(image_bytes).decode("utf-8")

    logger.info(
        "Sending image (%d bytes) to Anthropic Claude Vision for analysis.",
        len(image_bytes),
    )

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=_ANALYSIS_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            # Claude Vision expects the image as a base64
                            # source block with an explicit media type.
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyse this plant image and diagnose any disease.",
                        },
                    ],
                }
            ],
        )
    except Exception:
        logger.exception("Anthropic Claude Vision API call failed.")
        raise RuntimeError("Cloud fallback failed — Anthropic API returned an error.")

    # Parse the JSON text block from Claude's response.
    raw_text: str = response.content[0].text.strip()
    return _parse_cloud_response(raw_text, provider="Anthropic")


async def _analyze_with_gemini(image_bytes: bytes, media_type: str) -> dict:
    """Send the image to Google Gemini and return structured JSON.

    The new google-genai SDK is synchronous, so we offload the blocking
    call to a background thread with asyncio.to_thread().  This keeps
    FastAPI's event loop free to handle other requests while we wait for
    Gemini.

    Think of it like asking a coworker to make a phone call for you —
    you hand them the task and continue your own work until they tap you
    on the shoulder with the answer.
    """
    import asyncio

    client = _get_gemini_client()

    logger.info(
        "Sending image (%d bytes) to Google Gemini for analysis.",
        len(image_bytes),
    )

    def _sync_gemini_call() -> str:
        """Synchronous Gemini API call — runs in a thread."""
        from google.genai import types
        from PIL import Image

        # The new SDK accepts PIL Image objects directly in the contents
        # list alongside plain text — no manual base64 encoding needed.
        pil_image = Image.open(io.BytesIO(image_bytes))

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                _ANALYSIS_PROMPT + "\n\nAnalyse this plant image and diagnose any disease.",
                pil_image,
            ],
            config=types.GenerateContentConfig(
                # Forces raw JSON output — no markdown fences to strip.
                response_mime_type="application/json",
            ),
        )
        return response.text

    try:
        raw_text: str = await asyncio.to_thread(_sync_gemini_call)
    except Exception:
        logger.exception("Google Gemini API call failed.")
        raise RuntimeError("Cloud fallback failed — Gemini API returned an error.")

    return _parse_cloud_response(raw_text, provider="Gemini")


# ──────────────────────────────────────────────
# Shared response parser
# ──────────────────────────────────────────────

def _parse_cloud_response(raw_text: str, provider: str) -> dict:
    """Parse and validate JSON from either cloud provider.

    Both providers are instructed to return the same schema, so one
    parser handles both.  This is the single point of validation —
    if a provider misbehaves, we catch it here.

    Parameters
    ----------
    raw_text : str
        The raw text response from the cloud API.
    provider : str
        Name of the provider (for logging).

    Returns
    -------
    dict
        Validated result with keys: disease_detected, disease_name,
        confidence, recommended_action.
    """
    try:
        result: dict = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.error("%s returned non-JSON response: %.200s", provider, raw_text)
        raise RuntimeError(
            f"Cloud fallback ({provider}) returned an unparseable response. "
            "Check the prompt or API model version."
        )

    # Validate required keys are present.
    required_keys = ("disease_detected", "disease_name", "confidence", "recommended_action")
    for key in required_keys:
        if key not in result:
            logger.error(
                "%s response missing required key '%s': %s", provider, key, result
            )
            raise RuntimeError(
                f"Cloud fallback ({provider}) response missing '{key}' field."
            )

    # Normalise types for downstream consistency.
    result["disease_detected"] = bool(result["disease_detected"])
    result["disease_name"] = str(result["disease_name"])
    result["confidence"] = str(result["confidence"])
    result["recommended_action"] = str(result["recommended_action"])

    logger.info(
        "Cloud fallback (%s) → disease_detected=%s, disease_name='%s', confidence='%s'",
        provider,
        result["disease_detected"],
        result["disease_name"],
        result["confidence"],
    )

    return result


# ──────────────────────────────────────────────
# Public API — single entry point for main.py
# ──────────────────────────────────────────────

async def analyze_with_cloud(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
) -> dict:
    """Send a plant image to the available cloud vision API for diagnosis.

    The function checks which API key is present in the environment and
    routes to the corresponding provider.  Only one key needs to be set.

    Cascade fallback strategy
    ─────────────────────────
    Unlike simple priority routing (try A, ignore B), this function
    implements a **cascade**: if the primary provider fails at runtime
    (bad key, network error, malformed response), it automatically
    falls through to the secondary provider before giving up.

    Think of it like calling two doctors: if the first one doesn't
    answer the phone, you try the second one before telling the
    patient "no doctors available."

    Parameters
    ----------
    image_bytes : bytes
        Raw image file contents (JPEG or PNG).
    media_type : str
        MIME type of the image (default ``"image/jpeg"``).

    Returns
    -------
    dict
        ``{"disease_detected": bool, "disease_name": str,
          "confidence": str, "recommended_action": str}``

    Raises
    ------
    RuntimeError
        If ALL configured providers fail or no keys are set.
    """
    # ── Build an ordered list of providers to try ──
    # Each entry is a (name, async_callable) tuple.  We try them in
    # priority order and stop at the first success.  This replaces the
    # old if/return pattern that would crash on the first failure
    # without ever trying the next provider.
    providers: list[tuple[str, object]] = []

    if ANTHROPIC_API_KEY:
        providers.append(("Anthropic Claude Vision", _analyze_with_anthropic))
    if GEMINI_API_KEY:
        providers.append(("Google Gemini", _analyze_with_gemini))

    if not providers:
        # Neither key is configured — return a graceful error dict so
        # the server doesn't crash.  main.py can check disease_detected
        # =False and confidence="low" to know the fallback was a no-op.
        logger.warning(
            "No cloud API key found (checked ANTHROPIC_API_KEY and GEMINI_API_KEY). "
            "Cloud fallback is unavailable."
        )
        raise RuntimeError(
            "Cloud fallback unavailable — no API key configured. "
            "Set ANTHROPIC_API_KEY or GEMINI_API_KEY in your .env file."
        )

    # ── Cascade through providers ──
    # Try each provider in order.  If one fails (bad key, network error,
    # malformed response), log the error and fall through to the next.
    # Only raise if ALL providers fail.
    last_error: Optional[Exception] = None

    for provider_name, provider_fn in providers:
        try:
            logger.info("Cloud fallback routing to %s.", provider_name)
            return await provider_fn(image_bytes, media_type)
        except (ValueError, RuntimeError) as exc:
            # This provider failed — log it and try the next one.
            logger.warning(
                "Cloud provider '%s' failed: %s. Trying next provider...",
                provider_name, exc,
            )
            last_error = exc

    # All providers exhausted — raise the last error so main.py can
    # fall back to the local prediction gracefully.
    raise RuntimeError(
        f"All cloud providers failed. Last error: {last_error}"
    )
