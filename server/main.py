"""
main.py — FastAPI Application (The "Brain")

This is the central router for the SIP Plant Health Monitoring System.
It wires together:
  • Image ingestion  → local TFLite inference → cloud fallback (if needed)
  • Moisture ingestion → watering decision logic
  • Real-time WebSocket broadcasting to the dashboard
  • Static file serving for the dashboard and saved images

Design notes
────────────
• A `lifespan` context manager replaces the deprecated `@app.on_event`
  decorators (CLAUDE.md §6 FastAPI Conventions).
• All synchronous / blocking work (file writes, DB calls) is offloaded
  to a threadpool via `asyncio.to_thread()` so we never stall the ASGI
  event loop (CLAUDE.md §6 Async & Blocking I/O).
• The confidence threshold that gates local-vs-cloud AI is read from
  `.env` — not hardcoded (CLAUDE.md §5).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.inference import load_model, is_model_loaded, run_inference
from server.cloud_fallback import analyze_with_cloud
from server.alert_manager import manager
from server.data_logger import create_tables, log_analysis

# ──────────────────────────────────────────────
# Environment & logging
# ──────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
)

# Read config from .env (CLAUDE.md §5 — never hardcode these).
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
IMAGE_SAVE_DIR: str = os.getenv("IMAGE_SAVE_DIR", "saved_images/")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# Soil moisture thresholds — read from .env as floats.
# Below LOW → pump ON.  Above HIGH → pump OFF.  Between → hold state.
MOISTURE_LOW_THRESHOLD: float = float(os.getenv("MOISTURE_LOW_THRESHOLD", "30"))
MOISTURE_HIGH_THRESHOLD: float = float(os.getenv("MOISTURE_HIGH_THRESHOLD", "70"))

# JPEG magic bytes — the first two bytes of every valid JPEG file.
# Used to verify uploads are genuine JPEGs, not just renamed .txt files.
_JPEG_MAGIC: bytes = b"\xff\xd8"


# ──────────────────────────────────────────────
# Helpers (synchronous — always called via to_thread)
# ──────────────────────────────────────────────
def _ensure_image_dir() -> None:
    """Create the image-save directory if it doesn't exist yet.

    Uses a relative path read from .env so it maps cleanly to a Docker
    volume mount later (CLAUDE.md §6 Docker rules).
    """
    Path(IMAGE_SAVE_DIR).mkdir(parents=True, exist_ok=True)


def _save_image_to_disk(image_bytes: bytes, filename: str) -> str:
    """Persist a JPEG to IMAGE_SAVE_DIR and return the relative path.

    Why save to disk?  The dashboard needs to display thumbnails.
    FastAPI serves them from the mounted `/images` static route.
    """
    save_path = Path(IMAGE_SAVE_DIR) / filename
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    logger.info("Image saved → %s (%d bytes)", save_path, len(image_bytes))
    return str(save_path)


# ──────────────────────────────────────────────
# Lifespan — runs once at startup & shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown hook using the modern lifespan pattern.

    Why lifespan instead of @app.on_event?
    FastAPI deprecated on_event in favour of lifespan context managers.
    The lifespan pattern also lets us cleanly release resources on
    shutdown (e.g., close DB connections) inside the `finally` / after
    the `yield`.
    """
    # ── Startup ──────────────────────────────
    logger.info("=== SIP Plant Monitor — starting up ===")

    # 1. Ensure the image-save directory exists.
    _ensure_image_dir()

    # 2. Initialise the SQLite database — creates tables on first run,
    #    no-ops on subsequent boots (safe to call every startup).
    create_tables()

    # 3. Load the TFLite model.  If the model file is missing, the server
    #    still starts (for moisture-only mode) but image inference returns 503.
    model_ok = load_model()
    if not model_ok:
        logger.warning(
            "TFLite model unavailable — /ingest/image will rely on cloud fallback only."
        )

    yield  # ← Server is running and accepting requests here.

    # ── Shutdown ─────────────────────────────
    logger.info("=== SIP Plant Monitor — shutting down ===")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="SIP Plant Monitor System",
    lifespan=lifespan,
)

# Mount the dashboard UI (plain HTML/CSS/JS).
app.mount("/static", StaticFiles(directory="server/static"), name="static")

# Mount saved images so the dashboard can display thumbnails.
# The directory is created at startup by _ensure_image_dir().
# We mount after the dir exists to avoid a startup crash.
_ensure_image_dir()  # Safe to call twice — mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_SAVE_DIR), name="images")


# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────
class MoistureData(BaseModel):
    """Schema for incoming soil moisture readings from the sensor / simulator."""
    value: float


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/")
async def root() -> dict:
    """Health-check endpoint — confirms the server is alive."""
    return {"status": "online", "message": "Plant Monitor Server is running"}


@app.post("/ingest/image")
async def receive_image(file: UploadFile = File(...)) -> dict:
    """Receive a JPEG from the ESP32-CAM and run the hybrid AI pipeline.

    Pipeline (CLAUDE.md §2):
      1. Validate the upload is a genuine JPEG.
      2. Save the image to disk (non-blocking).
      3. Run local TFLite inference.
      4. If confidence < threshold → escalate to Claude Vision API.
      5. Broadcast result to dashboard via WebSocket.
      6. Return the final prediction.
    """
    # ── Step 1: Read & validate ──────────────
    image_bytes: bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Check JPEG magic bytes — more reliable than trusting the filename
    # extension, which an ESP32 may not even set (CLAUDE.md §6 Security).
    if not image_bytes[:2] == _JPEG_MAGIC:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG images are accepted.",
        )

    # ── Step 2: Save image (non-blocking) ────
    # Generate a unique filename to avoid collisions from concurrent uploads.
    filename = f"{uuid.uuid4().hex}.jpg"

    # Offload synchronous disk I/O to a threadpool so we don't block the
    # ASGI event loop (CLAUDE.md §6 Async & Blocking I/O).
    await asyncio.to_thread(_save_image_to_disk, image_bytes, filename)

    # ── Step 3: Local TFLite inference ───────
    # run_inference now returns a BINARY result:
    #   {"status": "Healthy"|"Diseased", "confidence": float,
    #    "top_class": str, "healthy_prob": float, "diseased_prob": float}
    source: str = "local"  # Track which AI produced the result

    if is_model_loaded():
        try:
            # run_inference is synchronous (NumPy / TFLite), so offload it.
            result: dict = await asyncio.to_thread(run_inference, image_bytes)
        except Exception:
            logger.exception("Local inference failed for '%s'.", filename)
            # If local inference crashes, fall through to cloud fallback
            # rather than returning a 500 — resilience over rigidity.
            result = {"status": "Diseased", "confidence": 0.0,
                      "top_class": "error", "healthy_prob": 0.0, "diseased_prob": 0.0}
    else:
        # Model not loaded (file missing) — skip straight to cloud.
        logger.info("No local model available; forwarding directly to cloud fallback.")
        result = {"status": "Diseased", "confidence": 0.0,
                  "top_class": "unknown", "healthy_prob": 0.0, "diseased_prob": 0.0}

    # ── Step 4: Threshold gate ───────────────
    # If the local model's aggregated binary confidence is high enough,
    # accept its verdict.  Otherwise, escalate to Claude for a second
    # opinion.  The threshold (default 0.75) is read from .env.
    if result["confidence"] >= CONFIDENCE_THRESHOLD:
        # Local model is confident — accept its binary prediction.
        logger.info(
            ">>> LOCAL AI ACCEPTED — status='%s', confidence=%.4f "
            "(healthy=%.4f, diseased=%.4f, threshold=%.2f)",
            result["status"], result["confidence"],
            result.get("healthy_prob", 0.0), result.get("diseased_prob", 0.0),
            CONFIDENCE_THRESHOLD,
        )
    else:
        # Local model is uncertain — neither bucket has a decisive lead.
        # Escalate to Claude Vision for expert analysis.
        logger.info(
            "Local confidence %.4f < threshold %.2f — escalating to cloud fallback.",
            result["confidence"],
            CONFIDENCE_THRESHOLD,
        )
        try:
            cloud_result = await analyze_with_cloud(image_bytes)
            source = "cloud"

            # Cloud fallback returns a unified schema from either Anthropic
            # or Gemini: {"disease_detected": bool, "disease_name": str,
            # "confidence": str, "recommended_action": str}.
            # Map into our binary status schema for dashboard consistency.
            is_cloud_healthy = not cloud_result.get("disease_detected", False)

            # Cloud confidence is a string ("high"/"medium"/"low").
            # Map to a numeric value so the dashboard can display it.
            confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.50}
            cloud_conf_str = cloud_result.get("confidence", "low").lower()
            cloud_conf_num = confidence_map.get(cloud_conf_str, 0.50)

            result = {
                "status": "Healthy" if is_cloud_healthy else "Diseased",
                "confidence": cloud_conf_num,
                "top_class": cloud_result.get("disease_name", "unknown"),
                "description": cloud_result.get("recommended_action", ""),
                "healthy_prob": cloud_conf_num if is_cloud_healthy else 0.0,
                "diseased_prob": cloud_conf_num if not is_cloud_healthy else 0.0,
            }

            logger.info(
                ">>> CLOUD FALLBACK TRIGGERED — status='%s', label='%s', confidence=%.4f",
                result["status"], result["top_class"], result["confidence"],
            )
        except (ValueError, RuntimeError) as exc:
            # Cloud fallback failed — return the local result as best-effort
            # rather than crashing.  The dashboard will show the low confidence.
            logger.error("Cloud fallback unavailable: %s", exc)
            source = "local (cloud unavailable)"

    # ── Step 5: Build response & broadcast ───
    # The response now uses the binary "status" key ("Healthy" / "Diseased")
    # instead of raw PlantVillage label names.  The original top_class is
    # still included for transparency and debugging.
    status = result.get("status", "Diseased")
    confidence = result.get("confidence", 0.0)
    disease_detected = (status == "Diseased")

    response = {
        "filename": filename,
        "image_url": f"/images/{filename}",
        "source": source,
        "status": status,
        "confidence": round(confidence, 4),
        "top_class": result.get("top_class", "unknown"),
        "description": result.get("description", ""),
        "is_disease": disease_detected,
        "healthy_prob": result.get("healthy_prob", 0.0),
        "diseased_prob": result.get("diseased_prob", 0.0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Broadcast to all connected dashboard clients via WebSocket.
    # This is what makes the dashboard update in real-time without polling.
    await manager.broadcast_alert({"type": "image_result", **response})

    # ── Step 6: Log to database (non-blocking) ──
    # Persist every result — local or cloud — to SQLite so we have a
    # permanent audit trail.  Offloaded to a threadpool because
    # SQLAlchemy is synchronous (CLAUDE.md §6 Async & Blocking I/O).
    try:
        await asyncio.to_thread(
            log_analysis,
            filename=filename,
            image_url=f"/images/{filename}",
            label=status,
            confidence=confidence,
            source=source,
            is_disease=disease_detected,
            description=result.get("description", ""),
            timestamp=response["timestamp"],
        )
    except Exception:
        # DB failure must not crash the response — the prediction is
        # still valid and has been broadcast.  Log and move on.
        logger.exception("Failed to persist analysis result to database.")

    if disease_detected:
        logger.warning(
            "DISEASE ALERT — status='%s', confidence=%.4f, top_class='%s', file='%s'",
            status, confidence, result.get("top_class", ""), filename,
        )

    return response


@app.post("/ingest/moisture")
async def receive_moisture(data: MoistureData) -> dict:
    """Receive a soil moisture reading from the sensor / simulator.

    Watering logic (CLAUDE.md §5 thresholds):
      • If moisture < MOISTURE_LOW_THRESHOLD (30%)  → pump ON
      • If moisture >= MOISTURE_LOW_THRESHOLD (30%) → pump OFF

    The comparison is simple: soil too dry → water it.  Both the
    incoming value and the threshold are cast to float explicitly to
    prevent type-mismatch bugs (e.g., if the sensor sends a string).
    """
    # Cast both sides to float to guarantee a numeric comparison.
    # This prevents subtle bugs where one side is a string or int.
    moisture_value: float = float(data.value)

    # Core decision: is the soil too dry?
    # Below the low threshold → pump should activate.
    water_command: bool = moisture_value < MOISTURE_LOW_THRESHOLD

    logger.info(
        "Moisture reading: %.2f%% (threshold=%.1f%%) → water=%s",
        moisture_value, MOISTURE_LOW_THRESHOLD, water_command,
    )

    moisture_payload = {
        "type": "moisture",
        "value": data.value,
        "water": water_command,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Broadcast the moisture reading to all dashboard clients so the
    # gauge updates in real-time.
    await manager.broadcast_alert(moisture_payload)

    return {"water": water_command, "received_value": data.value}


# ──────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Persistent WebSocket connection for dashboard clients.

    How it works (radio tower analogy):
      1. The browser opens a WebSocket to ws://host:port/ws.
      2. We accept and register the connection in the manager.
      3. We enter an infinite receive loop that keeps the connection
         alive.  The browser doesn't send meaningful data — the loop
         just prevents the socket from closing.
      4. Meanwhile, POST /ingest/image and /ingest/moisture call
         manager.broadcast_alert(), which pushes JSON to every
         registered socket — including this one.
      5. When the browser tab closes, `receive_text()` raises
         WebSocketDisconnect, and we clean up.
    """
    await manager.connect(websocket)
    try:
        # Keep the connection alive by waiting for messages.
        # The dashboard's JS sends periodic pings; we just consume them.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ──────────────────────────────────────────────
# Direct-run entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    # Always bind to 0.0.0.0 so the server is reachable from inside a
    # Docker container (CLAUDE.md §6 Docker rules).
    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=True)
