/**
 * app.js — SIP Plant Monitor Dashboard (Vanilla JS)
 *
 * Connects to the FastAPI WebSocket at ws://<host>/ws and updates the
 * DOM in real-time when the backend broadcasts analysis results and
 * moisture readings.
 *
 * BACKEND PAYLOAD REFERENCE (image_result):
 * {
 *   "type":       "image_result",                                    // added by broadcast_alert()
 *   "filename":   "f7d909eb83fc4180ba61cd1858066987.jpg",
 *   "image_url":  "/images/f7d909eb83fc4180ba61cd1858066987.jpg",
 *   "source":     "local",                                           // "local" or "cloud"
 *   "status":     "Healthy",                                         // "Healthy" or "Diseased"
 *   "confidence": 0.9609375,                                         // float 0.0–1.0
 *   "description":"",
 *   "is_disease":  false                                              // boolean
 * }
 *
 * BACKEND PAYLOAD REFERENCE (moisture):
 * {
 *   "type":      "moisture",
 *   "value":     42.5,
 *   "water":     false,
 *   "timestamp": "2026-04-13T..."
 * }
 *
 * HTML ID CONTRACT — every getElementById below MUST have a matching
 * id="..." in index.html.  If you rename one, rename the other.
 *
 * No frameworks — vanilla DOM only (CLAUDE.md §3 Frontend).
 */

(function () {
    "use strict";

    // ──────────────────────────────────────────────
    // 1. DOM REFERENCES
    // ──────────────────────────────────────────────
    // Grab every element we need to update, ONCE, at page load.
    // If any of these return null the dashboard is broken — the
    // console.log guard below will tell you which one.

    var dom = {
        // Top banner
        banner:       document.getElementById("status-banner"),
        statusIcon:   document.getElementById("status-icon"),
        statusText:   document.getElementById("status-text"),
        wsIndicator:  document.getElementById("ws-indicator"),

        // Left column — Latest Analysis panel
        latestImage:    document.getElementById("latest-image"),       // <img>
        placeholder:    document.getElementById("image-placeholder"),  // "No image yet" <p>
        analysisStatus: document.getElementById("analysis-status"),    // "Healthy" / "Diseased"
        analysisConf:   document.getElementById("analysis-confidence"),// "96.1%"
        analysisSource: document.getElementById("analysis-source"),    // "Local" / "Cloud"

        // Centre column — Moisture gauge
        gaugeFill:    document.getElementById("gauge-fill"),
        gaugeValue:   document.getElementById("gauge-value"),
        waterStatus:  document.getElementById("water-status"),
        moistureTime: document.getElementById("moisture-time"),

        // Right column — Alert feed
        alertFeed:  document.getElementById("alert-feed"),
        alertCount: document.getElementById("alert-count"),
    };

    // Startup sanity check — if any element is missing, log it
    // immediately so you don't chase silent null-reference crashes.
    Object.keys(dom).forEach(function (key) {
        if (dom[key] === null) {
            console.error("[INIT] dom." + key + " is NULL — check the id in index.html");
        }
    });

    var alertTotal = 0;
    var MAX_FEED_ITEMS = 50;


    // ──────────────────────────────────────────────
    // 2. WEBSOCKET CONNECTION
    // ──────────────────────────────────────────────

    function getWsUrl() {
        var proto = (window.location.protocol === "https:") ? "wss:" : "ws:";
        return proto + "//" + window.location.host + "/ws";
    }

    var ws = null;
    var reconnectDelay = 1000;
    var MAX_RECONNECT_DELAY = 30000;
    var pingInterval = null;

    function connect() {
        ws = new WebSocket(getWsUrl());

        ws.addEventListener("open", function () {
            console.log("[WS] Connected to", getWsUrl());
            reconnectDelay = 1000;
            setWsStatus(true);

            // Keep-alive ping every 25 s to survive proxy idle timeouts.
            clearInterval(pingInterval);
            pingInterval = setInterval(function () {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("ping");
                }
            }, 25000);
        });

        // ─────────────────────────────────────────
        // THE MESSAGE HANDLER — this is where every
        // WebSocket payload from the server is processed.
        // ─────────────────────────────────────────
        ws.addEventListener("message", function (event) {
            // Step 1: Parse the JSON string into an object.
            var data;
            try {
                data = JSON.parse(event.data);
            } catch (err) {
                console.warn("[WS] Received non-JSON:", event.data);
                return;
            }

            // Step 2: Log the raw object so you can inspect it in
            // Browser DevTools → Console tab.  This is your debug lifeline.
            console.log("WS Received:", data);

            // Step 3: Route to the correct handler.
            // The backend adds a "type" key during broadcast_alert().
            // We also handle the edge case where "type" might be missing
            // by checking for known payload fields as a fallback.
            if (data.type === "image_result" || data.status !== undefined) {
                handleImageResult(data);
            } else if (data.type === "moisture" || data.value !== undefined) {
                handleMoistureReading(data);
            } else {
                console.warn("[WS] Unknown message shape:", data);
            }
        });

        ws.addEventListener("close", function () {
            console.log("[WS] Disconnected — will retry in", reconnectDelay, "ms");
            setWsStatus(false);
            clearInterval(pingInterval);
            setTimeout(function () {
                reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
                connect();
            }, reconnectDelay);
        });

        ws.addEventListener("error", function () {
            console.error("[WS] Connection error");
        });
    }

    function setWsStatus(connected) {
        dom.wsIndicator.textContent = connected ? "WS: Connected" : "WS: Disconnected";
        dom.wsIndicator.className = connected
            ? "ws-badge ws-badge--connected"
            : "ws-badge ws-badge--disconnected";
    }


    // ──────────────────────────────────────────────
    // 3. IMAGE RESULT HANDLER
    // ──────────────────────────────────────────────
    // Called when the backend broadcasts a plant analysis result.
    //
    // Expected payload fields (from your exact backend output):
    //   data.image_url   → "/images/abc123.jpg"   → set as <img> src
    //   data.status      → "Healthy" or "Diseased" → display in Status card
    //   data.confidence  → 0.9609375              → format as "96.1%"
    //   data.source      → "local" or "cloud"     → capitalize → "Local"
    //   data.is_disease  → false                  → controls colour coding

    function handleImageResult(data) {
        // ── A. Update the image thumbnail ───────────
        // Set the <img> src to the saved JPEG path so the photo appears.
        // Then add the "visible" CSS class to flip it from display:none
        // to display:block (the CSS starts it hidden).
        dom.latestImage.src = data.image_url;
        dom.latestImage.alt = data.status + " (" + formatConfidence(data.confidence) + ")";
        dom.latestImage.classList.add("visible");

        // Hide the "No image received yet" placeholder text.
        dom.placeholder.classList.add("hidden");

        // ── B. Update the Status card ───────────────
        // Write "Healthy" or "Diseased" into id="analysis-status".
        dom.analysisStatus.textContent = data.status;

        // Colour it: green for healthy, red for diseased.
        if (data.is_disease) {
            dom.analysisStatus.className = "meta-card__value meta-card__value--alert";
        } else {
            dom.analysisStatus.className = "meta-card__value meta-card__value--healthy";
        }

        // ── C. Update the Confidence card ───────────
        // Convert the float (e.g. 0.9609375) to "96.1%".
        dom.analysisConf.textContent = formatConfidence(data.confidence);

        // ── D. Update the Source card ────────────────
        // Capitalize "local" → "Local", "cloud" → "Cloud".
        dom.analysisSource.textContent = capitalize(data.source);

        // ── E. Update the top banner ────────────────
        if (data.is_disease) {
            dom.banner.className = "banner banner--alert";
            dom.statusText.textContent = "Disease Alert \u2014 Check Dashboard";
        } else {
            dom.banner.className = "banner banner--healthy";
            dom.statusText.textContent = "System Healthy";
        }

        // ── F. Add an entry to the alert feed ───────
        addAlertEntry(data);
    }


    // ──────────────────────────────────────────────
    // 4. MOISTURE READING HANDLER
    // ──────────────────────────────────────────────

    function handleMoistureReading(data) {
        var pct = Math.max(0, Math.min(100, data.value));

        // Update gauge bar height.
        dom.gaugeFill.style.height = pct + "%";

        // Colour the bar: red=dry, green=ok, blue=wet.
        var level;
        if (pct < 30) {
            level = "low";
        } else if (pct > 70) {
            level = "high";
        } else {
            level = "mid";
        }
        dom.gaugeFill.setAttribute("data-level", level);

        // Text readouts.
        dom.gaugeValue.textContent = pct.toFixed(1) + "%";
        dom.moistureTime.textContent = formatTime(data.timestamp);

        // Watering status.
        if (data.water) {
            dom.waterStatus.textContent = "ACTIVE";
            dom.waterStatus.className = "meta-card__value meta-card__value--alert";
        } else {
            dom.waterStatus.textContent = "Idle";
            dom.waterStatus.className = "meta-card__value meta-card__value--healthy";
        }

        // Add to alert feed.
        addAlertEntry(data);
    }


    // ──────────────────────────────────────────────
    // 5. ALERT FEED
    // ──────────────────────────────────────────────

    function addAlertEntry(data) {
        // Remove the "Waiting for data…" placeholder on first entry.
        var empty = dom.alertFeed.querySelector(".alert-feed__empty");
        if (empty) { empty.remove(); }

        var li = document.createElement("li");

        if (data.type === "image_result" || data.status !== undefined) {
            // Image analysis entry.
            li.className = "alert-item " +
                (data.is_disease ? "alert-item--disease" : "alert-item--healthy");
            li.innerHTML =
                '<div class="alert-item__time">' + formatTime(data.timestamp) + "</div>" +
                '<div class="alert-item__body">' +
                    escapeHtml(data.status) + " \u2014 " +
                    formatConfidence(data.confidence) +
                "</div>" +
                '<div class="alert-item__detail">' +
                    "Source: " + capitalize(data.source) +
                "</div>";
        } else {
            // Moisture reading entry.
            li.className = "alert-item alert-item--moisture";
            li.innerHTML =
                '<div class="alert-item__time">' + formatTime(data.timestamp) + "</div>" +
                '<div class="alert-item__body">' +
                    "Moisture: " + data.value.toFixed(1) + "%" +
                "</div>" +
                '<div class="alert-item__detail">' +
                    "Watering: " + (data.water ? "ACTIVE" : "Idle") +
                "</div>";
        }

        dom.alertFeed.prepend(li);

        alertTotal++;
        dom.alertCount.textContent = alertTotal;

        // Cap feed length to prevent memory growth.
        while (dom.alertFeed.children.length > MAX_FEED_ITEMS) {
            dom.alertFeed.removeChild(dom.alertFeed.lastChild);
        }
    }


    // ──────────────────────────────────────────────
    // 6. UTILITY FUNCTIONS
    // ──────────────────────────────────────────────

    /**
     * Convert a float like 0.9609375 into the string "96.1%".
     * Handles undefined/null gracefully (returns "—").
     */
    function formatConfidence(value) {
        if (value === undefined || value === null) { return "\u2014"; }
        return (value * 100).toFixed(1) + "%";
    }

    /**
     * Capitalize the first letter of a string.
     * "local" → "Local",  "cloud" → "Cloud".
     * Handles undefined/null gracefully (returns "—").
     */
    function capitalize(str) {
        if (!str) { return "\u2014"; }
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    /** Format an ISO timestamp into a short local time string. */
    function formatTime(isoString) {
        if (!isoString) { return "\u2014"; }
        try {
            return new Date(isoString).toLocaleTimeString(
                [], { hour: "2-digit", minute: "2-digit", second: "2-digit" }
            );
        } catch (e) {
            return isoString;
        }
    }

    /** Escape HTML to prevent XSS (CLAUDE.md §6 Security). */
    function escapeHtml(text) {
        if (!text) { return ""; }
        var el = document.createElement("span");
        el.textContent = text;
        return el.innerHTML;
    }


    // ──────────────────────────────────────────────
    // 7. BOOT
    // ──────────────────────────────────────────────
    connect();

})();
