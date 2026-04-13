"""
mock_device.py — ESP32-CAM & Sensor Simulator

Simulates the two data streams that the real hardware will produce:
  1. JPEG images   → POST /ingest/image
  2. Moisture data → POST /ingest/moisture

Why a simulator?
────────────────
Building the full software pipeline *before* the hardware arrives means
we can test, debug, and iterate without soldering anything.  This script
replaces the ESP32-CAM + soil sensor with synthetic data that exercises
the exact same API endpoints.

Modes
─────
  --mode healthy    High-confidence "healthy" images (solid green JPEG).
  --mode diseased   High-confidence "diseased" images (brown-spotted JPEG).
  --mode ambiguous  Low-confidence images (noisy JPEG) that should trigger
                    the cloud fallback path.
  --mode real       Picks a random real image from data/plantvillage_samples/.
                    The folder name is the true class label, which is logged
                    before sending so you can compare against the model's
                    prediction.

Usage
─────
  python -m simulator.mock_device --mode healthy --interval 5
  python -m simulator.mock_device --mode real --interval 5
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | mock_device | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
IMAGE_ENDPOINT = f"{BASE_URL}/ingest/image"
MOISTURE_ENDPOINT = f"{BASE_URL}/ingest/moisture"

# Synthetic JPEG dimensions (small — we're simulating an ESP32-CAM
# whose typical output is 640×480, but for a POC 64×64 is enough to
# exercise the full pipeline without wasting bandwidth).
IMG_SIZE = (64, 64)


# ──────────────────────────────────────────────
# Synthetic image generators
# ──────────────────────────────────────────────
# Each mode produces a visually distinct image so you can tell at a
# glance on the dashboard which "plant" the simulator is pretending to be.

def _generate_healthy_image() -> bytes:
    """Create a bright-green image representing a healthy leaf.

    A solid green square with a few darker-green circles to mimic
    natural leaf texture.
    """
    img = Image.new("RGB", IMG_SIZE, color=(34, 139, 34))  # Forest green
    draw = ImageDraw.Draw(img)
    # Add subtle "vein" circles
    for _ in range(5):
        x = random.randint(5, IMG_SIZE[0] - 5)
        y = random.randint(5, IMG_SIZE[1] - 5)
        r = random.randint(2, 6)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 100, 0))
    return _pil_to_jpeg_bytes(img)


def _generate_diseased_image() -> bytes:
    """Create a green image with brown spots representing disease.

    Brown circles on a green background simulate common fungal
    leaf-spot diseases visible to both TFLite and Claude Vision.
    """
    img = Image.new("RGB", IMG_SIZE, color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    # Add "disease spots" — brown circles scattered across the leaf
    for _ in range(8):
        x = random.randint(8, IMG_SIZE[0] - 8)
        y = random.randint(8, IMG_SIZE[1] - 8)
        r = random.randint(3, 8)
        # Brownish spot with slight colour variation
        brown = (
            random.randint(100, 140),
            random.randint(60, 80),
            random.randint(20, 40),
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=brown)
    return _pil_to_jpeg_bytes(img)


def _generate_ambiguous_image() -> bytes:
    """Create a noisy, blurry image that should produce low confidence.

    This exercises the cloud-fallback path: the local TFLite model
    should return confidence < 0.80, triggering escalation to Claude.
    Random pixel noise makes the image genuinely hard to classify.
    """
    # Fill every pixel with random colours — pure noise.
    img = Image.new("RGB", IMG_SIZE)
    pixels = img.load()
    for x in range(IMG_SIZE[0]):
        for y in range(IMG_SIZE[1]):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
    return _pil_to_jpeg_bytes(img)


def _pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    """Convert a Pillow Image to JPEG bytes in memory.

    Why JPEG?  The real ESP32-CAM outputs JPEG natively, so the
    server pipeline is built around that format.  Using JPEG here
    ensures the simulator exercises the same code paths.
    """
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# Map CLI mode names to generator functions (synthetic modes only).
# The "real" mode uses _pick_real_image() instead — see run_simulator().
_GENERATORS = {
    "healthy": _generate_healthy_image,
    "diseased": _generate_diseased_image,
    "ambiguous": _generate_ambiguous_image,
}

# ──────────────────────────────────────────────
# Real dataset image picker
# ──────────────────────────────────────────────
# Default path to the PlantVillage subset — relative to the project root.
_DEFAULT_DATASET_DIR = "data/plantvillage_samples"


def _discover_dataset(dataset_dir: str) -> dict[str, list[Path]]:
    """Scan the dataset directory and build a map of class → image paths.

    The PlantVillage layout is:
      data/plantvillage_samples/
        Potato___Early_blight/
          Potato___Early_blight_0.jpg
          ...
        Strawberry___Leaf_scorch/
          ...

    Each subfolder name IS the ground-truth class label.  We collect all
    .jpg files under each subfolder so we can pick one at random later.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        logger.error("Dataset directory not found: '%s'", dataset_dir)
        sys.exit(1)

    class_map: dict[str, list[Path]] = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        # Gather all JPEG files in this class folder.
        images = list(class_dir.glob("*.jpg"))
        if images:
            class_map[class_dir.name] = images

    if not class_map:
        logger.error("No class folders with .jpg files found in '%s'.", dataset_dir)
        sys.exit(1)

    total_images = sum(len(v) for v in class_map.values())
    logger.info(
        "Dataset loaded — %d classes, %d total images from '%s'",
        len(class_map), total_images, dataset_dir,
    )
    return class_map


def _pick_real_image(class_map: dict[str, list[Path]]) -> tuple[bytes, str, str]:
    """Randomly select a real image from the dataset.

    Returns
    -------
    tuple[bytes, str, str]
        (jpeg_bytes, true_class_label, filename)
    """
    # Pick a random class, then a random image within that class.
    true_class = random.choice(list(class_map.keys()))
    image_path = random.choice(class_map[true_class])

    with open(image_path, "rb") as f:
        jpeg_bytes = f.read()

    return jpeg_bytes, true_class, image_path.name


# ──────────────────────────────────────────────
# Moisture simulator
# ──────────────────────────────────────────────
def _generate_moisture(mode: str) -> float:
    """Return a simulated soil moisture percentage (0–100).

    The value is biased by mode so the watering logic gets a
    realistic range of inputs:
      • healthy   → well-watered soil (50–80%)
      • diseased  → overwatered (75–100%) — excess moisture promotes disease
      • ambiguous → random (10–90%)
      • real      → full range (10–90%) since the dataset has mixed classes
    """
    ranges = {
        "healthy": (50.0, 80.0),
        "diseased": (75.0, 100.0),
        "ambiguous": (10.0, 90.0),
        "real": (10.0, 90.0),
    }
    low, high = ranges[mode]
    return round(random.uniform(low, high), 2)


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
def run_simulator(mode: str, interval: float, dataset_dir: str) -> None:
    """Infinite loop that POSTs image data to the FastAPI server.

    In 'real' mode, images come from the PlantVillage dataset on disk.
    In synthetic modes (healthy/diseased/ambiguous), images are generated
    in-memory with Pillow.

    Each iteration sends one image and one moisture reading, then
    sleeps for `interval` seconds.  Ctrl-C stops it cleanly.
    """
    # For "real" mode, pre-scan the dataset directory once so we don't
    # hit the filesystem on every cycle.  For synthetic modes, grab the
    # generator function from the lookup table.
    class_map: dict[str, list[Path]] | None = None
    generate_image = None

    if mode == "real":
        class_map = _discover_dataset(dataset_dir)
    else:
        generate_image = _GENERATORS[mode]

    logger.info(
        "Starting simulator — mode=%s, interval=%.1fs, target=%s",
        mode, interval, BASE_URL,
    )

    cycle = 0
    while True:
        cycle += 1
        logger.info("── Cycle %d ──", cycle)

        # ── Send image ───────────────────────
        try:
            if mode == "real":
                # Pick a random real image and log its true class so we
                # can compare against whatever the model predicts.
                jpeg_bytes, true_class, orig_filename = _pick_real_image(class_map)
                logger.info(
                    "SELECTED IMAGE — file='%s', true_class='%s'",
                    orig_filename, true_class,
                )
            else:
                jpeg_bytes = generate_image()
                orig_filename = f"plant_{cycle}.jpg"

            # `files` dict matches what FastAPI's UploadFile expects via
            # python-multipart: field name "file", filename, content type.
            files = {
                "file": (orig_filename, jpeg_bytes, "image/jpeg"),
            }
            resp = requests.post(IMAGE_ENDPOINT, files=files, timeout=30)
            resp.raise_for_status()
            logger.info("Image  → %s", resp.json())
        except requests.ConnectionError:
            logger.error("Cannot connect to %s — is the server running?", BASE_URL)
        except Exception:
            logger.exception("Image POST failed.")

        # ── Send moisture ────────────────────
        try:
            moisture = _generate_moisture(mode)
            resp = requests.post(
                MOISTURE_ENDPOINT,
                json={"value": moisture},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Moisture → %.2f%% | response: %s", moisture, resp.json())
        except requests.ConnectionError:
            logger.error("Cannot connect to %s — is the server running?", BASE_URL)
        except Exception:
            logger.exception("Moisture POST failed.")

        # ── Wait before next cycle ───────────
        logger.info("Sleeping %.1fs...", interval)
        time.sleep(interval)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simulator."""
    parser = argparse.ArgumentParser(
        description="Simulate ESP32-CAM + soil sensor data for the SIP Plant Monitor.",
    )
    parser.add_argument(
        "--mode",
        choices=["healthy", "diseased", "ambiguous", "real"],
        default="healthy",
        help=(
            "Type of data to send. 'real' picks random images from the "
            "PlantVillage dataset on disk (default: healthy)."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between each data cycle (default: 5.0).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=_DEFAULT_DATASET_DIR,
        help=(
            "Path to the PlantVillage dataset directory. Only used with "
            "--mode real (default: data/plantvillage_samples)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_simulator(
            mode=args.mode,
            interval=args.interval,
            dataset_dir=args.dataset_dir,
        )
    except KeyboardInterrupt:
        logger.info("Simulator stopped by user (Ctrl-C).")
        sys.exit(0)
