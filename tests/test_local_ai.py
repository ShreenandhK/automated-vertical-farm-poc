"""
test_local_ai.py — Binary Classification Verification Script

A standalone utility that loads sample images from data/plantvillage_samples/,
runs them through the TFLite model via inference.py, and prints the
**aggregated binary result** (Healthy vs. Diseased) alongside the raw
probability breakdown.

What this verifies:
  1. The TFLite model loads from the path in .env.
  2. Preprocessing (resize to 64×64, int8 quantisation) works correctly.
  3. Dequantised Softmax probabilities are properly aggregated into two
     buckets: healthy_prob and diseased_prob.
  4. The confidence threshold gate (0.65) routes correctly to LOCAL or CLOUD.

How aggregation solves confidence splitting:
──────────────────────────────────────────────
The model has 33 output classes.  Several of these are "healthy" variants
(Apple___healthy, Tomato___healthy, Grape___healthy, etc.).  When the
model sees a healthy leaf, it distributes probability across ALL healthy
classes — not just the correct plant's healthy class.  For example:

  Tomato___healthy   = 0.25       ← argmax picks this (only 0.25!)
  Apple___healthy    = 0.18
  Potato___healthy   = 0.15
  Grape___healthy    = 0.10
  ...other healthy   = 0.07
  ─────────────────────────
  TOTAL healthy      = 0.75       ← aggregation recovers the true signal

Without aggregation, the top class (0.25) would fail a 0.65 threshold
and trigger an unnecessary (and costly) cloud fallback.  By summing,
we see the model is 75% sure it's healthy — well above 0.65.

Usage:
    python -m tests.test_local_ai

    Run from the project root (SIP-project/) so relative paths resolve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Setup — load environment before importing inference
# ──────────────────────────────────────────────
load_dotenv()

# Add project root to sys.path so `from server.inference import ...` works
# when running this script directly with `python tests/test_local_ai.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.inference import load_model, run_inference, is_model_loaded  # noqa: E402

# The confidence threshold from .env — same value the server uses to
# decide local-vs-cloud routing.
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# Directory containing sample images organised by class subfolder.
SAMPLES_DIR = PROJECT_ROOT / "data" / "plantvillage_samples"


def main() -> None:
    """Load the model, iterate over sample images, and print binary predictions."""

    print("=" * 90)
    print("  SIP Plant Monitor — Binary Classification Test (Healthy vs. Diseased)")
    print("=" * 90)
    print()

    # ── Step 1: Load the TFLite model ────────
    print("[1/3] Loading TFLite model...")
    if not load_model():
        print("  ✗ Model failed to load. Check TFLITE_MODEL_PATH in .env")
        print(f"    Current path: {os.getenv('TFLITE_MODEL_PATH', '(not set)')}")
        sys.exit(1)
    print("  ✓ Model loaded successfully.\n")

    # ── Step 2: Discover sample images ───────
    print(f"[2/3] Scanning samples directory: {SAMPLES_DIR}\n")

    if not SAMPLES_DIR.exists():
        print(f"  ✗ Samples directory not found: {SAMPLES_DIR}")
        sys.exit(1)

    # Collect all JPEG files, grouped by class subfolder.
    # Structure: data/plantvillage_samples/<ClassName>/<image>.jpg
    image_files: list[tuple[str, Path]] = []
    for class_dir in sorted(SAMPLES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.jpg"))[:2]:
            # Take up to 2 images per class to keep output readable.
            image_files.append((class_dir.name, img_path))

    if not image_files:
        print("  ✗ No .jpg files found in sample subdirectories.")
        sys.exit(1)

    print(f"  Found {len(image_files)} sample image(s) across "
          f"{len(set(c for c, _ in image_files))} class(es).\n")

    # ── Step 3: Run inference on each image ──
    print("[3/3] Running binary inference...\n")
    print(f"  {'FOLDER (expected)':<35} {'STATUS':<10} {'CONF':>5}  "
          f"{'HLTH':>5} {'DIS':>5}  {'TOP CLASS':<35} ROUTE   MATCH")
    print(f"  {'-'*35} {'-'*10} {'-'*5}  {'-'*5} {'-'*5}  {'-'*35} {'-'*7} {'-'*5}")

    correct = 0
    total = 0
    local_count = 0
    cloud_count = 0

    for expected_class, img_path in image_files:
        # Read the raw JPEG bytes — exactly what the ESP32-CAM would send.
        image_bytes = img_path.read_bytes()

        try:
            result = run_inference(image_bytes)
        except Exception as exc:
            print(f"  {expected_class:<35} ERROR: {exc}")
            continue

        status = result["status"]
        confidence = result["confidence"]
        healthy_prob = result["healthy_prob"]
        diseased_prob = result["diseased_prob"]
        top_class = result["top_class"]

        # Determine how the threshold gate would route this image.
        if confidence >= CONFIDENCE_THRESHOLD:
            route = "LOCAL"
            local_count += 1
        else:
            route = "CLOUD"
            cloud_count += 1

        # Determine the EXPECTED binary status from the folder name.
        # If the folder name contains "healthy", we expect "Healthy".
        expected_status = "Healthy" if "healthy" in expected_class.lower() else "Diseased"

        # Check if the binary prediction matches the expected status.
        match = (status == expected_status)
        if match:
            correct += 1
        total += 1

        indicator = "✓" if match else "✗"
        print(f"  {expected_class:<35} {status:<10} {confidence:>5.2f}  "
              f"{healthy_prob:>5.2f} {diseased_prob:>5.2f}  {top_class:<35} {route:<7} {indicator}")

    # ── Summary ──────────────────────────────
    print()
    print("=" * 90)
    print(f"  Binary accuracy (Healthy vs. Diseased): {correct}/{total} "
          f"({(correct / total * 100) if total else 0:.1f}%)")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Images routed to LOCAL:  {local_count}")
    print(f"  Images routed to CLOUD:  {cloud_count}")
    print()
    print("  Note: HLTH = aggregated healthy probability, DIS = aggregated diseased probability.")
    print("  The higher bucket wins and becomes the confidence score.")
    print("=" * 90)


if __name__ == "__main__":
    main()
