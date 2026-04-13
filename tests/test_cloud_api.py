"""
test_cloud_api.py — Dual-Provider Cloud Fallback Integration Test

Verifies that the cloud fallback pipeline in cloud_fallback.py works
end-to-end with a real plant image and a real API call.

Supports both providers:
  • Anthropic Claude (claude-sonnet-4-20250514)
  • Google Gemini   (gemini-2.5-flash)

At startup the script checks which API keys are present in .env:
  - If BOTH keys are found, the user is prompted to choose a provider.
  - If only ONE key is found, that provider is used automatically.
  - If NEITHER key is found, the script exits with a clear error.

What this tests:
  1. The chosen API key loads correctly from .env.
  2. A random image (healthy or diseased) is selected from
     data/plantvillage_samples/.
  3. The image is sent to the chosen cloud model using the same prompt
     and config used in cloud_fallback.py.
  4. The response is strict JSON (no markdown wrapping).
  5. The response matches the required schema:
       {"disease_detected", "disease_name", "confidence", "recommended_action"}

Why both healthy and diseased images?
───────────────────────────────────────
Cloud fallback can be triggered by any image the local model is uncertain
about — not just diseased ones.  A healthy plant photographed in poor
lighting can also fall below the confidence threshold.  Selecting randomly
from the full dataset verifies the cloud model handles both cases correctly.

Usage:
    python -m tests.test_cloud_api

    Run from the project root (SIP-project/) so relative .env and image
    paths resolve correctly.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Setup — resolve project root first, then load
# .env using an explicit path.
#
# Why explicit path?  load_dotenv() without arguments searches upward
# from the CWD which can miss the file depending on how Python is
# invoked.  Pinning to PROJECT_ROOT guarantees we always load the
# right .env regardless of working directory.
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# override=True ensures .env values win over any stale system-level
# environment variables that may have been set in a previous session.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

# ──────────────────────────────────────────────
# Paths & API keys
# ──────────────────────────────────────────────
SAMPLES_DIR = PROJECT_ROOT / "data" / "plantvillage_samples"

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# ──────────────────────────────────────────────
# Shared prompt — identical to cloud_fallback.py
# so this test validates exactly what the live system sends.
# ──────────────────────────────────────────────
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
# Image discovery
# ──────────────────────────────────────────────
def _find_all_images() -> list[Path]:
    """Collect all .jpg files from every class folder (healthy and diseased).

    The plantvillage_samples directory uses the naming convention:
      Plant___Disease/  (diseased)
      Plant___healthy/  (healthy)

    Both are included so the random selection can land on either type.
    """
    if not SAMPLES_DIR.exists():
        print(f"  ERROR: Samples directory not found: {SAMPLES_DIR}")
        print("  Make sure you are running from the project root.")
        sys.exit(1)

    all_images: list[Path] = []
    for class_dir in SAMPLES_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*.jpg"):
            all_images.append(img_path)
    return all_images


# ──────────────────────────────────────────────
# Response validation — shared by both providers
# ──────────────────────────────────────────────
def _validate_and_print(raw_text: str, file_name: str, elapsed_ms: float, provider: str) -> None:
    """Parse, pretty-print, and schema-validate the JSON response.

    Both providers must return the same schema, so one validator handles
    both — this mirrors the design of _parse_cloud_response() in
    cloud_fallback.py.
    """
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Provider      : {provider}")
    print(f"  Selected file : {file_name}")
    print(f"  API call time : {elapsed_ms:.0f} ms")
    print()
    print(f"  Raw JSON response from {provider}:")
    print("  " + "-" * 50)

    try:
        parsed: dict = json.loads(raw_text)
        pretty_json = json.dumps(parsed, indent=4)
        for line in pretty_json.splitlines():
            print(f"  {line}")
    except json.JSONDecodeError:
        print("  WARNING: Response was not valid JSON!")
        print(f"  Raw text: {raw_text}")
        sys.exit(1)

    print("  " + "-" * 50)
    print()

    # Validate required schema keys
    print("  Schema validation:")
    required_keys = ("disease_detected", "disease_name", "confidence", "recommended_action")
    all_ok = True
    for key in required_keys:
        present = key in parsed
        status = "OK     " if present else "MISSING"
        print(f"    [{status}] {key}")
        if not present:
            all_ok = False

    print()
    if all_ok:
        print("  All required keys present.")
        print(f"  disease_detected   : {parsed['disease_detected']}")
        print(f"  disease_name       : {parsed['disease_name']}")
        print(f"  confidence         : {parsed['confidence']}")
        print(f"  recommended_action : {parsed['recommended_action']}")
        print()
        print("=" * 70)
        print(f"  {provider} integration test PASSED.")
        print("=" * 70)
    else:
        print("  Schema mismatch — check the prompt or model response above.")
        sys.exit(1)


# ──────────────────────────────────────────────
# Provider: Anthropic Claude
# ──────────────────────────────────────────────
def _run_anthropic_test(image_bytes: bytes, file_name: str) -> None:
    """Send the image to Claude Vision and validate the response.

    Uses base64 encoding — Claude's vision API accepts images as base64
    source blocks with an explicit media type.  No temp file needed.
    """
    print("[3/4] Configuring Anthropic SDK and sending request...")

    try:
        import anthropic
    except ImportError as exc:
        print(f"  ERROR: Missing dependency — {exc}")
        print("  Run:  pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Encode the image bytes to base64 — Claude's required image format.
    image_b64: str = base64.b64encode(image_bytes).decode("utf-8")

    print("  Sending to claude-sonnet-4-20250514...\n")
    start_time = time.perf_counter()

    try:
        response = client.messages.create(
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
                                "media_type": "image/jpeg",
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
    except Exception as exc:
        print(f"  ERROR: Anthropic API call failed — {exc}")
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    raw_text: str = response.content[0].text.strip()
    _validate_and_print(raw_text, file_name, elapsed_ms, "Anthropic Claude")


# ──────────────────────────────────────────────
# Provider: Google Gemini
# ──────────────────────────────────────────────
def _run_gemini_test(image_bytes: bytes, file_name: str) -> None:
    """Send the image to Gemini and validate the response.

    Uses the new google-genai SDK with an explicit Client object and
    types.GenerateContentConfig for strict JSON output.
    """
    print("[3/4] Configuring Gemini SDK and sending request...")

    try:
        from google import genai
        from google.genai import types
        from PIL import Image
    except ImportError as exc:
        print(f"  ERROR: Missing dependency — {exc}")
        print("  Run:  pip install google-genai pillow")
        sys.exit(1)

    # The new google-genai SDK uses an explicit Client object rather than
    # a module-level configure() call — cleaner and easier to test.
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Convert raw bytes to a PIL Image — the new SDK accepts PIL objects
    # directly in the contents list.  No temp file, no disk I/O.
    pil_image = Image.open(io.BytesIO(image_bytes))

    print("  Sending to gemini-2.5-flash...\n")
    start_time = time.perf_counter()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                # Combine the system prompt and task in one text block,
                # followed by the PIL image.  The new SDK handles mixed
                # content lists (text + image) natively.
                _ANALYSIS_PROMPT + "\n\nAnalyse this plant image and diagnose any disease.",
                pil_image,
            ],
            config=types.GenerateContentConfig(
                # Forces raw JSON output — no markdown fences to strip.
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        print(f"  ERROR: Gemini API call failed — {exc}")
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    raw_text: str = response.text.strip()
    _validate_and_print(raw_text, file_name, elapsed_ms, "Google Gemini")


# ──────────────────────────────────────────────
# Provider selection
# ──────────────────────────────────────────────
def _select_provider() -> str:
    """Determine which provider to test based on available API keys.

    - Both keys present  → prompt the user to choose
    - Only one key       → use it automatically, no prompt needed
    - Neither key        → exit with a clear error
    """
    has_anthropic = bool(ANTHROPIC_API_KEY)
    has_gemini = bool(GEMINI_API_KEY)

    print("[1/4] Checking available API keys...")

    if has_anthropic:
        print(f"  ANTHROPIC_API_KEY : found (first 8 chars: {ANTHROPIC_API_KEY[:8]}...)")
    else:
        print("  ANTHROPIC_API_KEY : not set")

    if has_gemini:
        print(f"  GEMINI_API_KEY    : found (first 8 chars: {GEMINI_API_KEY[:8]}...)")
    else:
        print("  GEMINI_API_KEY    : not set")

    print()

    if has_anthropic and has_gemini:
        # Both keys available — ask the user which to test.
        print("  Both API keys are available. Which provider would you like to test?")
        print("    [1] Anthropic Claude  (claude-sonnet-4-20250514)")
        print("    [2] Google Gemini     (gemini-2.0-flash)")
        print()

        while True:
            choice = input("  Enter 1 or 2: ").strip()
            if choice == "1":
                print()
                return "anthropic"
            elif choice == "2":
                print()
                return "gemini"
            else:
                print("  Invalid choice — please enter 1 or 2.")

    elif has_anthropic:
        print("  Only ANTHROPIC_API_KEY found — using Anthropic Claude automatically.")
        print()
        return "anthropic"

    elif has_gemini:
        print("  Only GEMINI_API_KEY found — using Google Gemini automatically.")
        print()
        return "gemini"

    else:
        print("  ERROR: No API keys found in .env.")
        print("  Add at least one of the following to your .env file:")
        print("    ANTHROPIC_API_KEY=sk-ant-...")
        print("    GEMINI_API_KEY=AIza...")
        sys.exit(1)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    """Run the cloud API integration test for the selected provider."""

    print("=" * 70)
    print("  SIP Plant Monitor — Cloud API Integration Test")
    print("=" * 70)
    print()

    # ── Step 1: Detect keys & pick provider ──
    provider = _select_provider()

    # ── Step 2: Select a random image ────────
    print("[2/4] Selecting random image (healthy or diseased)...")
    all_images = _find_all_images()

    if not all_images:
        print(f"  ERROR: No .jpg images found under {SAMPLES_DIR}")
        sys.exit(1)

    # random.choice gives a different image every run — healthy or diseased.
    selected_image: Path = random.choice(all_images)
    image_bytes: bytes = selected_image.read_bytes()

    class_name = selected_image.parent.name
    file_name = selected_image.name

    print(f"  Class folder : {class_name}")
    print(f"  File name    : {file_name}")
    print(f"  Image size   : {len(image_bytes):,} bytes\n")

    # ── Steps 3 & 4: Call the chosen provider ─
    if provider == "anthropic":
        _run_anthropic_test(image_bytes, file_name)
    else:
        _run_gemini_test(image_bytes, file_name)


if __name__ == "__main__":
    main()
