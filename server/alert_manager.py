"""
alert_manager.py — WebSocket Connection Manager

Manages all active WebSocket connections from dashboard clients and
broadcasts real-time alerts when new data arrives from the pipeline.

Think of this like a radio tower:
  • Each browser tab that opens the dashboard "tunes in" (connects).
  • When the server has news (new image analysed, moisture reading,
    disease alert), it broadcasts to every tuned-in listener at once.
  • If a listener disconnects (closes the tab), we quietly remove them
    so the next broadcast doesn't crash on a dead socket.

Design decisions
────────────────
• A simple in-memory list of connections is fine for this POC.  In
  production you'd use Redis pub/sub or similar for multi-process
  support — but CLAUDE.md §6 says no Redis, and a single Uvicorn
  worker is sufficient for our scale.
• All methods are async because WebSocket send/receive are I/O-bound
  operations that must not block the event loop.
• The broadcast is fire-and-forget for each client: if one socket
  errors during send, we disconnect it and continue to the others.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Connection Manager
# ──────────────────────────────────────────────
class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts JSON messages.

    Usage (inside main.py):
        manager = ConnectionManager()

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await manager.connect(ws)
            try:
                while True:
                    await ws.receive_text()   # keep-alive
            except WebSocketDisconnect:
                manager.disconnect(ws)
    """

    def __init__(self) -> None:
        # Why a list and not a set?  WebSocket objects are not hashable
        # by default in Starlette, so a list is the simplest container.
        self._connections: list[WebSocket] = []

    # ── Connection lifecycle ─────────────────

    async def connect(self, websocket: WebSocket) -> None:
        """Accept an incoming WebSocket handshake and register the client.

        The `accept()` call completes the HTTP → WebSocket upgrade.
        Until we call it, the browser's `new WebSocket(url)` hangs.
        """
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(
            "WebSocket client connected. Total active: %d",
            len(self._connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a client that has disconnected (tab closed, network drop).

        This is synchronous because there's no I/O — we're just
        removing an item from the list.  Called from the except block
        in the WebSocket endpoint.
        """
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Total active: %d",
            len(self._connections),
        )

    @property
    def active_count(self) -> int:
        """Number of currently connected dashboard clients."""
        return len(self._connections)

    # ── Broadcasting ─────────────────────────

    async def broadcast_alert(self, data: dict[str, Any]) -> None:
        """Send a JSON message to every connected dashboard client.

        Parameters
        ----------
        data : dict
            Payload to broadcast.  Must be JSON-serialisable.
            Typical shapes:

            Image analysis result:
                {
                    "type": "image_result",
                    "filename": "abc123.jpg",
                    "image_url": "/images/abc123.jpg",
                    "label": "leaf_blight",
                    "confidence": 0.92,
                    "source": "cloud",
                    "is_disease": true,
                    "description": "..."
                }

            Moisture reading:
                {
                    "type": "moisture",
                    "value": 42.5,
                    "water": false
                }

        Dead connections are silently removed so the next broadcast
        doesn't waste time on unreachable clients.
        """
        if not self._connections:
            return  # No listeners — nothing to do.

        message: str = json.dumps(data)

        # Iterate over a copy of the list because we may remove items
        # mid-loop if a send fails (e.g., client disconnected between
        # the last receive and this broadcast).
        stale: list[WebSocket] = []

        for connection in self._connections:
            try:
                await connection.send_text(message)
            except Exception:
                # The client is gone — mark for removal.  We don't log
                # at ERROR level because disconnects are routine.
                logger.debug("Failed to send to a client; marking as stale.")
                stale.append(connection)

        # Clean up dead connections after the loop.
        for dead in stale:
            self.disconnect(dead)

        if stale:
            logger.info(
                "Removed %d stale WebSocket connection(s).", len(stale)
            )


# ──────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────
# A single shared instance used by main.py's routes and WS endpoint.
# This is safe because Uvicorn runs one event loop per worker, and our
# POC uses a single worker (CLAUDE.md §6 Statelessness — persistent
# state lives in SQLite, not in-memory).
manager = ConnectionManager()
