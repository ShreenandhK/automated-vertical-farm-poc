"""
inference.py — Local TFLite Binary Plant Health Classifier

This module wraps a generic TFLite image-classification model behind a
single public function: `run_inference(image_bytes) -> dict`.  The model
file and label file are read from environment variables so the classifier
can be swapped without touching code.

Binary aggregation strategy
───────────────────────────
The underlying TFLite model is a 33-class PlantVillage classifier whose
Softmax output assigns a probability to every specific class (e.g.,
"Tomato___healthy", "Apple___healthy", "Potato___Early_blight").

For our system we only need to answer ONE question: is this plant
**Healthy** or **Diseased**?  If we naively use `argmax`, a perfectly
healthy leaf might score only 0.25 on its top class because the
remaining "healthy" probability is *split* across Apple_healthy,
Grape_healthy, Peach_healthy, etc.  That 0.25 would fail the confidence
gate and unnecessarily escalate to Claude — even though the model was
80 %+ certain the plant is healthy overall.

The fix: after dequantising the 33 Softmax probabilities we **aggregate**
them into two buckets:
  • `healthy_prob`  — sum of all classes whose label contains "healthy"
  • `diseased_prob` — sum of everything else
The larger bucket wins, and its sum becomes the confidence score.  This
gives us a clean binary signal with confidence values that actually
reflect the model's *overall* certainty.

Design decisions
────────────────
• The TFLite interpreter is loaded **once** at module-import time and
  reused across requests.  Creating an interpreter per request would be
  far too slow for real-time image analysis from an ESP32-CAM.
• Pillow is used for preprocessing because it is already in our stack;
  NumPy is installed as a standalone dependency.
• The classifier is completely label-agnostic — labels are loaded from a
  plain-text file, one per line.  This keeps the AI model "generic" as
  required by CLAUDE.md §6 (AI Model Flexibility).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import io

from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Environment & logging
# ──────────────────────────────────────────────
load_dotenv()  # Read .env once at import time
logger = logging.getLogger(__name__)

# Base directory for resolving relative model/label paths.
# Uses this file's location (server/) so paths resolve correctly regardless
# of the working directory — critical when running inside Docker where cwd
# may differ from the project root.
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Paths come from .env — never hardcoded (CLAUDE.md §5 / §6 Docker rules).
# Defaults use os.path.join(BASE_DIR, ...) to build absolute paths from the
# server/ directory, avoiding breakage when cwd != project root.
TFLITE_MODEL_PATH: str = os.getenv(
    "TFLITE_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "plant_disease_model.tflite"),
)
TFLITE_LABELS_PATH: str = os.getenv(
    "TFLITE_LABELS_PATH",
    os.path.join(BASE_DIR, "models", "labels.txt"),
)


# ──────────────────────────────────────────────
# Label loader
# ──────────────────────────────────────────────
def _load_labels(path: str) -> list[str]:
    """Read class labels from a plain-text file (one label per line).

    Why a separate function?  So we can give a clear error message if the
    file is missing instead of crashing deep inside NumPy indexing.
    """
    label_path = Path(path)
    if not label_path.exists():
        logger.warning("Labels file not found at '%s'. Predictions will use numeric indices.", path)
        return []
    with open(label_path, "r", encoding="utf-8") as fh:
        # strip() removes trailing newlines / whitespace from each line
        return [line.strip() for line in fh if line.strip()]


# ──────────────────────────────────────────────
# TFLite interpreter singleton
# ──────────────────────────────────────────────
# We import ai_edge_litert inside load_model() rather than at the top so
# the rest of the module (labels, helpers) can still be unit-tested
# without the runtime installed.
_interpreter: Optional[object] = None  # Will be an ai_edge_litert Interpreter
_input_details: Optional[list] = None
_output_details: Optional[list] = None
_labels: list[str] = []


def load_model() -> bool:
    """Initialise the TFLite interpreter and label list.

    Returns True on success, False if the model file is missing or corrupt.
    Called once at server startup (via FastAPI lifespan).  Keeping this
    explicit rather than auto-loading at import time gives main.py control
    over *when* the heavy load happens and lets us return a 503 if it fails.
    """
    global _interpreter, _input_details, _output_details, _labels  # noqa: PLW0603

    model_path = Path(TFLITE_MODEL_PATH)
    if not model_path.exists():
        logger.error("TFLite model not found at '%s'. Local inference will be unavailable.", TFLITE_MODEL_PATH)
        return False

    try:
        # ai-edge-litert is a lightweight TFLite-only runtime that avoids
        # the heavy tensorflow-cpu dependency and its protobuf conflicts
        # (e.g. with the google-genai SDK).
        from ai_edge_litert.interpreter import Interpreter

        _interpreter = Interpreter(model_path=str(model_path))
        _interpreter.allocate_tensors()

        # Cache tensor metadata so we don't look it up on every request.
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

        _labels = _load_labels(TFLITE_LABELS_PATH)
        label_count = len(_labels) if _labels else "unknown"
        logger.info(
            "TFLite model loaded successfully. Input shape: %s | Labels: %s",
            _input_details[0]["shape"].tolist(),
            label_count,
        )
        return True

    except Exception:
        logger.exception("Failed to load TFLite model from '%s'.", TFLITE_MODEL_PATH)
        return False


def is_model_loaded() -> bool:
    """Quick check used by route handlers before attempting inference."""
    return _interpreter is not None


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw JPEG bytes into the NumPy tensor the model expects.

    Why Pillow?  It's already in our stack (CLAUDE.md §3) and avoids
    pulling in OpenCV, which would be overkill for simple resize + normalize.

    The function reads the model's expected input shape and quantization
    parameters at runtime so it works with *any* TFLite classification
    model — float32, uint8-quantized, or int8-quantized.  You can swap
    the .tflite file and this code adapts automatically.

    Quantization primer:
      Quantized models store weights and activations as integers (int8 or
      uint8) instead of float32 to reduce model size and speed up inference
      on edge devices like the ESP32.  The mapping between real (float)
      values and quantized (integer) values is:

        real_value = (quantized_value - zero_point) * scale

      So to feed a float pixel value into a quantized model, we reverse it:

        quantized_value = real_value / scale + zero_point
    """
    # The model's input tensor tells us exactly what size it wants.
    input_shape = _input_details[0]["shape"]        # e.g. [1, 64, 64, 3]
    height, width = int(input_shape[1]), int(input_shape[2])

    # Open the JPEG from raw bytes, convert to RGB (handles grayscale/RGBA),
    # and resize to the model's expected dimensions using high-quality resampling.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)

    # Start with float32 pixel values normalized to [0.0, 1.0].
    # This is the "real value" that we'll either use directly (float model)
    # or quantize into the model's integer format.
    img_array = np.array(image, dtype=np.float32) / 255.0

    input_dtype = _input_details[0]["dtype"]

    if input_dtype == np.float32:
        # Float model — normalised [0, 1] is exactly what it expects.
        pass
    else:
        # Quantized model (int8 or uint8).  We need to convert our
        # normalised float pixels into the quantized integer range using
        # the model's stored scale and zero_point.
        quant_params = _input_details[0]["quantization_parameters"]
        scale = float(quant_params["scales"][0])
        zero_point = int(quant_params["zero_points"][0])

        # real_value = (q - zero_point) * scale  →  q = real / scale + zero_point
        img_array = (img_array / scale + zero_point).astype(input_dtype)

        logger.debug(
            "Quantized input: scale=%.6f, zero_point=%d, dtype=%s",
            scale, zero_point, input_dtype,
        )

    # Add the batch dimension: (H, W, 3) → (1, H, W, 3)
    return np.expand_dims(img_array, axis=0)


# ──────────────────────────────────────────────
# Internal: raw model execution
# ──────────────────────────────────────────────
def _run_model(image_bytes: bytes) -> np.ndarray:
    """Execute the TFLite model and return the dequantised probability array.

    This is a low-level helper that handles preprocessing, interpreter
    invocation, and int8 dequantisation.  The public `run_inference()`
    function builds on top of this to produce the final binary result.

    Returns
    -------
    np.ndarray
        A 1-D float32 array of Softmax probabilities, one per class.
    """
    # 1. Preprocess — resize and normalize the image to match model input.
    input_tensor = _preprocess_image(image_bytes)

    # 2. Feed the tensor into the interpreter and invoke inference.
    _interpreter.set_tensor(_input_details[0]["index"], input_tensor)
    _interpreter.invoke()

    # 3. Read the output — a 1-D array of Softmax probabilities.
    #    For quantized models, the raw tensor contains integers that must
    #    be dequantized back to float probabilities using:
    #      real_value = (quantized_value - zero_point) * scale
    output_data = _interpreter.get_tensor(_output_details[0]["index"])
    raw_output: np.ndarray = output_data[0]  # Remove batch dimension

    output_dtype = _output_details[0]["dtype"]
    if output_dtype != np.float32:
        # Dequantize: convert integer output back to real probabilities.
        quant_params = _output_details[0]["quantization_parameters"]
        scale = float(quant_params["scales"][0])
        zero_point = int(quant_params["zero_points"][0])
        probabilities = (raw_output.astype(np.float32) - zero_point) * scale
        logger.debug(
            "Dequantized output: scale=%.6f, zero_point=%d, range=[%.4f, %.4f]",
            scale, zero_point, probabilities.min(), probabilities.max(),
        )
    else:
        probabilities = raw_output.astype(np.float32)

    return probabilities


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def run_inference(image_bytes: bytes) -> dict:
    """Run a single image through the local TFLite model.

    Instead of returning the single top class from 33 PlantVillage
    labels, this function **aggregates** all Softmax probabilities into
    two buckets — Healthy vs. Diseased — and returns the winner.

    Why aggregate?
    ──────────────
    The model's Softmax distributes probability across 33 classes.  A
    healthy tomato leaf might score:
      Tomato___healthy   = 0.25
      Apple___healthy    = 0.18
      Potato___healthy   = 0.15
      …other healthy…   = 0.12
      ────────────────────────
      TOTAL healthy      = 0.70   ← the model IS fairly sure it's healthy
      TOTAL diseased     = 0.30

    With naive `argmax`, the top class is Tomato___healthy at 0.25 —
    far below any reasonable threshold.  By summing the buckets we
    recover the model's true overall certainty.

    Parameters
    ----------
    image_bytes : bytes
        Raw JPEG (or PNG) file contents as read from the upload.

    Returns
    -------
    dict
        {"status": str, "confidence": float, "top_class": str,
         "healthy_prob": float, "diseased_prob": float}

        - `status` is either ``"Healthy"`` or ``"Diseased"``.
        - `confidence` is the aggregated probability for the winning
          bucket (0.0–1.0).
        - `top_class` is the single highest-scoring label from the
          original 33 classes (useful for debugging / cloud context).
        - `healthy_prob` / `diseased_prob` are the raw bucket sums
          so the caller or dashboard can display the breakdown.

    Raises
    ------
    RuntimeError
        If the model has not been loaded yet (call `load_model()` first).
    """
    if not is_model_loaded():
        raise RuntimeError(
            "TFLite model is not loaded. Call load_model() during server startup."
        )

    # 1. Get the raw 33-class probability array from the model.
    probabilities = _run_model(image_bytes)

    # 2. Aggregate into two buckets: Healthy vs. Diseased.
    #    We iterate over every class probability and its label string.
    #    If the label contains "healthy" (case-insensitive), its
    #    probability goes into the healthy bucket; otherwise, diseased.
    healthy_prob: float = 0.0
    diseased_prob: float = 0.0

    for i, prob in enumerate(probabilities):
        # Look up the label string for this index.
        if _labels and i < len(_labels):
            label_name = _labels[i]
        else:
            label_name = f"class_{i}"

        # Route this class's probability to the correct bucket.
        if "healthy" in label_name.lower():
            healthy_prob += float(prob)
        else:
            diseased_prob += float(prob)

    # 3. Determine the winner — whichever bucket has more total
    #    probability is the model's overall verdict.
    if healthy_prob > diseased_prob:
        status = "Healthy"
        confidence = healthy_prob
    else:
        status = "Diseased"
        confidence = diseased_prob

    # 4. Also identify the single top class (for debugging / logs).
    #    This is the old argmax behaviour — kept for transparency.
    top_index = int(np.argmax(probabilities))
    if _labels and top_index < len(_labels):
        top_class = _labels[top_index]
    else:
        top_class = f"class_{top_index}"

    logger.info(
        "Local inference → status='%s', confidence=%.4f "
        "(healthy=%.4f, diseased=%.4f, top_class='%s')",
        status, confidence, healthy_prob, diseased_prob, top_class,
    )

    return {
        "status": status,
        "confidence": confidence,
        "top_class": top_class,
        "healthy_prob": round(healthy_prob, 4),
        "diseased_prob": round(diseased_prob, 4),
    }
