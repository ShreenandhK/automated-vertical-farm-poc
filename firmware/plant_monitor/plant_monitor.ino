/*
 * ============================================================================
 *  plant_monitor.ino — ESP32 Firmware for SIP Vertical Farm Monitor
 * ============================================================================
 *
 *  WHAT THIS DOES:
 *    1. Connects to your local WiFi network.
 *    2. Reads soil moisture (analog) and temperature (DHT22) on a timer.
 *    3. Displays live telemetry on a 128x64 OLED screen.
 *    4. POSTs telemetry as JSON to the Python/FastAPI server.
 *    5. Parses the server's response and controls a relay-driven water pump.
 *    6. (Future) Captures JPEG images and POSTs them for AI disease detection.
 *
 *  HARDWARE:
 *    - ESP32 DevKit v1 (or any standard ESP32 board)
 *    - Analog Soil Moisture Sensor  → GPIO 34
 *    - DHT22 Temperature Sensor     → GPIO 4 (with 10kΩ pull-up)
 *    - SSD1306 OLED 128x64 (I2C)   → SDA=GPIO 21, SCL=GPIO 22
 *    - 5V Relay Module              → GPIO 26
 *    - Submersible Water Pump       → wired through the relay
 *
 *  SERVER ENDPOINTS (must match your FastAPI backend):
 *    POST /ingest/moisture   — JSON body: {"value": <float>}
 *                              Response:   {"water": true/false, "received_value": <float>}
 *    POST /ingest/image      — Multipart JPEG (field name: "file")
 *                              Response:   {...analysis result...}
 *
 *  NON-BLOCKING DESIGN:
 *    This firmware uses millis()-based timing instead of delay().
 *    Think of it like checking your watch periodically rather than
 *    setting an alarm and freezing until it rings. This keeps the
 *    ESP32 responsive — it can update the OLED, respond to serial
 *    commands, and handle WiFi events even between sensor readings.
 *
 *  AUTHOR:  SIP Project — Phase 7 Hardware Integration
 *  LICENSE: MIT
 * ============================================================================
 */

// ─────────────────────────────────────────────────────────────
// LIBRARY INCLUDES
// ─────────────────────────────────────────────────────────────
// WiFi and HTTP — built into the ESP32 Arduino core
#include <WiFi.h>
#include <HTTPClient.h>

// JSON serialization/deserialization — ArduinoJson v7
// (Install via Library Manager: "ArduinoJson" by Benoit Blanchon)
#include <ArduinoJson.h>

// DHT temperature/humidity sensor
// (Install via Library Manager: "DHT sensor library" by Adafruit)
#include <DHT.h>

// OLED display — I2C SSD1306
// (Install via Library Manager: "Adafruit SSD1306" — it will also
//  prompt you to install "Adafruit GFX Library", accept that too)
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>


// ─────────────────────────────────────────────────────────────
// CONFIGURATION — EDIT THESE TO MATCH YOUR SETUP
// ─────────────────────────────────────────────────────────────

// WiFi credentials — replace with your actual network name and password.
// WARNING: Never commit real credentials to version control.
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Server URL — the IP address of the machine running your FastAPI server.
// Find it by running `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
// on the machine where `uvicorn server.main:app` is running.
// IMPORTANT: The ESP32 and server MUST be on the same local network.
const char* SERVER_BASE_URL = "http://192.168.1.100:8000";

// How often to read sensors and send data (in milliseconds).
// 10 seconds is a good default — fast enough for demos, slow enough
// to avoid flooding the server (your simulator used 5s).
const unsigned long TELEMETRY_INTERVAL_MS = 10000;  // 10 seconds

// How long to keep the pump running when the server says "water: true".
// This matches the WATERING_DURATION_SECONDS=5 in your .env file.
// The server decides IF we should water; the firmware decides HOW LONG.
const unsigned long PUMP_DURATION_MS = 5000;  // 5 seconds

// WiFi reconnection attempt interval — if WiFi drops, retry every 5 seconds.
// We don't want to spam reconnection attempts too fast.
const unsigned long WIFI_RETRY_INTERVAL_MS = 5000;


// ─────────────────────────────────────────────────────────────
// PIN DEFINITIONS
// ─────────────────────────────────────────────────────────────
// These must match the physical wiring from the Wiring Guide table.

// Soil Moisture Sensor — analog input.
// GPIO 34 is on ADC1 (safe to use with WiFi; ADC2 conflicts with WiFi!).
// The ESP32's ADC reads 0–4095 (12-bit resolution) for 0V–3.3V.
const int SOIL_MOISTURE_PIN = 34;

// DHT22 Temperature & Humidity Sensor — digital pin.
// GPIO 4 has no boot-strap conflicts and works well for digital I/O.
const int DHT_PIN  = 4;
const int DHT_TYPE = DHT22;  // Use DHT11 if you have that model instead

// Relay Module — digital output.
// GPIO 26 is a general-purpose output pin with no conflicts.
// Most relay modules are ACTIVE LOW: writing LOW energizes the relay (pump ON).
const int RELAY_PIN = 26;

// OLED Display — I2C pins (ESP32 defaults).
// These don't need to be defined explicitly — Wire.begin() uses
// GPIO 21 (SDA) and GPIO 22 (SCL) by default on the ESP32.
// But we define them for clarity.
const int OLED_SDA = 21;
const int OLED_SCL = 22;

// OLED dimensions.
const int SCREEN_WIDTH  = 128;
const int SCREEN_HEIGHT = 64;

// I2C address for most SSD1306 modules. If your display doesn't work,
// try 0x3D instead — some modules use that address.
const uint8_t OLED_I2C_ADDRESS = 0x3C;


// ─────────────────────────────────────────────────────────────
// SENSOR CALIBRATION
// ─────────────────────────────────────────────────────────────
// These values map the raw ADC reading to a 0–100% moisture scale.
//
// HOW TO CALIBRATE:
//   1. Read the sensor in DRY AIR     → that's your DRY_VALUE (high number).
//   2. Submerge the sensor in WATER   → that's your WET_VALUE (low number).
//   Note: Capacitive sensors read HIGH when dry, LOW when wet (inverse).
//   Your specific sensor may differ — adjust these after testing.
//
// Default values for a typical capacitive soil moisture sensor:
const int SOIL_DRY_VALUE = 3500;  // Raw ADC when sensor is in dry air
const int SOIL_WET_VALUE = 1500;  // Raw ADC when sensor is submerged in water


// ─────────────────────────────────────────────────────────────
// GLOBAL OBJECTS
// ─────────────────────────────────────────────────────────────

// DHT sensor object — handles the one-wire protocol internally.
DHT dht(DHT_PIN, DHT_TYPE);

// OLED display object — -1 means no hardware reset pin (most modules
// use an RC circuit for reset instead of a dedicated pin).
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);


// ─────────────────────────────────────────────────────────────
// STATE VARIABLES (non-blocking timing)
// ─────────────────────────────────────────────────────────────
// Instead of delay(), we track "when did we last do X?" and check
// if enough time has passed. This is the millis() pattern.
//
// Analogy: Imagine you're cooking. Instead of standing at the stove
// watching the timer (delay), you set a mental note and go do other
// things, glancing at the clock periodically (millis check).

unsigned long lastTelemetryTime   = 0;  // When we last read sensors + POSTed
unsigned long pumpStartTime       = 0;  // When the pump was turned ON
unsigned long lastWiFiRetryTime   = 0;  // When we last tried to reconnect WiFi
bool          pumpIsRunning       = false;  // Is the pump currently active?

// Latest sensor readings — stored globally so the OLED can display
// them at any time, not just right after a read cycle.
float currentMoisture    = 0.0;
float currentTemperature = 0.0;


// ─────────────────────────────────────────────────────────────
// FUNCTION DECLARATIONS (forward declarations for readability)
// ─────────────────────────────────────────────────────────────
void connectToWiFi();
float readSoilMoisture();
float readTemperature();
void  updateOLED(bool wifiConnected, float moisture, float temperature, bool pumpOn);
bool  postTelemetry(float moisture, float temperature);
void  activatePump();
void  deactivatePump();
void  captureAndPostImage();  // Stub for ESP32-CAM — commented out


// =============================================================
//  SETUP — Runs once when the ESP32 powers on or resets
// =============================================================
void setup() {
    // ── Serial Monitor ──────────────────────────
    // Start serial communication at 115200 baud so we can see debug
    // messages in the Arduino IDE Serial Monitor.
    Serial.begin(115200);
    while (!Serial) { ; }  // Wait for serial port to be ready
    Serial.println();
    Serial.println("============================================");
    Serial.println("  SIP Plant Monitor — ESP32 Firmware v1.0");
    Serial.println("============================================");

    // ── Pin Modes ───────────────────────────────
    // Configure GPIO directions. The soil sensor is on an ADC pin,
    // which defaults to input — but being explicit is good practice.
    pinMode(SOIL_MOISTURE_PIN, INPUT);

    // Relay pin must be OUTPUT so we can control it.
    // Start with the relay OFF (HIGH = off for active-low modules).
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, HIGH);  // Ensure pump is OFF at startup
    Serial.println("[INIT] Relay pin set to OUTPUT, pump OFF.");

    // ── DHT22 Sensor ────────────────────────────
    dht.begin();
    Serial.println("[INIT] DHT22 sensor initialized.");

    // ── OLED Display ────────────────────────────
    // Initialize I2C with our defined pins (in case they differ from
    // the ESP32 defaults — they don't here, but it's good practice).
    Wire.begin(OLED_SDA, OLED_SCL);

    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDRESS)) {
        // If the display fails to initialize, log the error but keep running.
        // The system is still useful without the display — it just won't show
        // local status. This follows the "degrade gracefully" principle.
        Serial.println("[ERROR] SSD1306 OLED initialization failed!");
        Serial.println("        Check wiring: SDA→GPIO21, SCL→GPIO22, VCC→3.3V");
    } else {
        Serial.println("[INIT] OLED display initialized (128x64, I2C).");
        // Show a boot splash for 2 seconds.
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 10);
        display.println("SIP Plant Monitor");
        display.setCursor(10, 30);
        display.println("Firmware v1.0");
        display.setCursor(10, 50);
        display.println("Connecting WiFi...");
        display.display();
    }

    // ── WiFi Connection ─────────────────────────
    connectToWiFi();

    Serial.println("[INIT] Setup complete. Entering main loop.");
    Serial.println("--------------------------------------------");
}


// =============================================================
//  LOOP — Runs continuously after setup()
// =============================================================
// This is the "heartbeat" of the firmware. Every iteration checks:
//   1. Is WiFi still connected? If not, try to reconnect.
//   2. Has enough time passed to read sensors? If yes, read + POST.
//   3. Is the pump running past its duration? If yes, turn it off.
//
// No delay() calls — everything is driven by millis() checks.
void loop() {
    unsigned long now = millis();

    // ── WiFi Health Check ───────────────────────
    // WiFi can drop unexpectedly (range, router restart, interference).
    // We check every loop iteration but only attempt reconnection
    // at the WIFI_RETRY_INTERVAL to avoid flooding.
    if (WiFi.status() != WL_CONNECTED) {
        if (now - lastWiFiRetryTime >= WIFI_RETRY_INTERVAL_MS) {
            Serial.println("[WIFI] Connection lost. Attempting reconnect...");
            connectToWiFi();
            lastWiFiRetryTime = now;
        }
    }

    // ── Telemetry Cycle ─────────────────────────
    // Read sensors and POST to the server at the configured interval.
    if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryTime = now;

        // Step 1: Read the sensors.
        currentMoisture    = readSoilMoisture();
        currentTemperature = readTemperature();

        // Step 2: Log readings to Serial Monitor for debugging.
        Serial.println("──── Telemetry Cycle ────");
        Serial.printf("  Moisture:    %.1f%%\n", currentMoisture);
        Serial.printf("  Temperature: %.1f°C\n", currentTemperature);

        // Step 3: Update the OLED with the latest readings.
        updateOLED(
            WiFi.status() == WL_CONNECTED,
            currentMoisture,
            currentTemperature,
            pumpIsRunning
        );

        // Step 4: POST telemetry to the server (only if WiFi is up).
        if (WiFi.status() == WL_CONNECTED) {
            bool shouldWater = postTelemetry(currentMoisture, currentTemperature);

            // Step 5: If the server says "water: true" and the pump
            // isn't already running, start the pump.
            if (shouldWater && !pumpIsRunning) {
                activatePump();
            }
        } else {
            Serial.println("  [SKIP] No WiFi — telemetry not sent.");
        }

        Serial.println("─────────────────────────");
    }

    // ── Pump Auto-Shutoff ───────────────────────
    // Safety: The pump runs for a fixed duration (PUMP_DURATION_MS),
    // then automatically shuts off. This prevents overwatering even
    // if the server keeps saying "water: true" — the firmware acts
    // as the last line of defense.
    //
    // Analogy: Like a kitchen timer that turns off the oven automatically.
    // The recipe (server) says "cook for 5 min", the timer (firmware)
    // enforces it mechanically.
    if (pumpIsRunning && (now - pumpStartTime >= PUMP_DURATION_MS)) {
        deactivatePump();
    }

    // No delay() here — the loop runs as fast as possible.
    // The millis() checks above gate when work actually happens.
    // A tiny yield to the WiFi/system task scheduler:
    yield();
}


// =============================================================
//  WiFi CONNECTION
// =============================================================
void connectToWiFi() {
    Serial.printf("[WIFI] Connecting to \"%s\"", WIFI_SSID);

    // WiFi.mode(WIFI_STA) sets the ESP32 as a "station" (client).
    // The alternative is WIFI_AP (access point) or WIFI_AP_STA (both).
    // We only need station mode — connect to an existing router.
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    // Wait up to 15 seconds for the connection to establish.
    // We use a timeout here (not in the main loop) because WiFi
    // is a prerequisite — there's nothing useful to do without it
    // during initial boot. After boot, reconnection is non-blocking.
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);  // Acceptable during setup — not in loop()
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println(" Connected!");
        Serial.printf("[WIFI] IP Address: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("[WIFI] Signal Strength (RSSI): %d dBm\n", WiFi.RSSI());
    } else {
        Serial.println(" FAILED!");
        Serial.println("[WIFI] Could not connect. Will retry in main loop.");
        Serial.println("        → Check SSID and password.");
        Serial.println("        → Ensure the router is within range.");
    }
}


// =============================================================
//  SOIL MOISTURE READING
// =============================================================
float readSoilMoisture() {
    // Read the raw analog value from the soil moisture sensor.
    // The ESP32 ADC returns 0–4095 (12-bit) for 0V–3.3V input.
    //
    // IMPORTANT: Capacitive sensors output HIGHER voltage when DRY
    // and LOWER voltage when WET. So the mapping is INVERTED:
    //   - High ADC value → dry soil → low moisture %
    //   - Low ADC value  → wet soil → high moisture %
    //
    // We take multiple readings and average them to reduce noise.
    // Analog sensors are inherently noisy — averaging smooths out
    // random spikes. Think of it like taking a patient's temperature
    // three times and using the average.

    const int NUM_SAMPLES = 10;
    long sum = 0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        sum += analogRead(SOIL_MOISTURE_PIN);
        // Small delay between reads to let the ADC settle.
        // This is a microsecond-level delay — acceptable even in loop().
        delayMicroseconds(100);
    }

    int rawValue = sum / NUM_SAMPLES;

    // Map the raw ADC value to a 0–100% scale.
    // constrain() clamps the value to the calibration range first,
    // then map() does the linear interpolation.
    //
    //   DRY_VALUE (3500) → 0%   (dry soil)
    //   WET_VALUE (1500) → 100% (saturated soil)
    //
    // Notice the map() parameters are INVERTED (dry→0, wet→100)
    // because higher ADC = drier soil.
    int constrained = constrain(rawValue, SOIL_WET_VALUE, SOIL_DRY_VALUE);
    float percentage = map(constrained, SOIL_DRY_VALUE, SOIL_WET_VALUE, 0, 100);

    // Clamp to 0–100 range as a safety net (map can return out-of-range
    // values if the input somehow exceeds the constrained bounds due
    // to integer rounding).
    percentage = constrain(percentage, 0.0f, 100.0f);

    Serial.printf("  [SOIL] Raw ADC: %d → Moisture: %.1f%%\n", rawValue, percentage);
    return percentage;
}


// =============================================================
//  TEMPERATURE READING
// =============================================================
float readTemperature() {
    // The DHT22 uses a proprietary one-wire protocol (not Dallas 1-Wire).
    // The Adafruit library handles all the timing-critical bit-banging.
    //
    // readTemperature() returns degrees Celsius by default.
    // Pass `true` as an argument to get Fahrenheit instead.
    float tempC = dht.readTemperature();

    // The DHT22 returns NaN (Not a Number) if the read fails.
    // Common causes: loose wiring, missing pull-up resistor, or
    // reading too fast (DHT22 needs ~2 seconds between reads).
    if (isnan(tempC)) {
        Serial.println("  [DHT] ERROR: Failed to read temperature!");
        Serial.println("         → Check wiring and 10kΩ pull-up resistor.");
        return -999.0;  // Sentinel value so the server knows it's invalid
    }

    Serial.printf("  [DHT] Temperature: %.1f°C\n", tempC);
    return tempC;
}


// =============================================================
//  OLED DISPLAY UPDATE
// =============================================================
void updateOLED(bool wifiConnected, float moisture, float temperature, bool pumpOn) {
    // Clear the entire display buffer before writing new content.
    // The SSD1306 library works with an in-memory buffer — you write
    // to the buffer, then call display() to push it to the screen.
    //
    // Analogy: Like a whiteboard. Erase it, write the new info,
    // then hold it up for everyone to see.
    display.clearDisplay();
    display.setTextSize(1);       // 6x8 pixels per character
    display.setTextColor(SSD1306_WHITE);

    // ── Row 1: WiFi Status (y=0) ────────────────
    display.setCursor(0, 0);
    if (wifiConnected) {
        display.printf("WiFi: Connected");
    } else {
        display.printf("WiFi: DISCONNECTED");
    }

    // ── Row 2: IP Address (y=12) ────────────────
    display.setCursor(0, 12);
    if (wifiConnected) {
        display.printf("IP: %s", WiFi.localIP().toString().c_str());
    } else {
        display.printf("IP: ---.---.---.---");
    }

    // ── Row 3: Moisture (y=28) ──────────────────
    display.setCursor(0, 28);
    display.printf("Moisture: %.1f%%", moisture);

    // ── Row 4: Temperature (y=40) ───────────────
    display.setCursor(0, 40);
    if (temperature > -999.0) {
        display.printf("Temp: %.1f C", temperature);
    } else {
        display.printf("Temp: ERROR");
    }

    // ── Row 5: Pump Status (y=54) ───────────────
    display.setCursor(0, 54);
    if (pumpOn) {
        display.printf("Pump: ON  [WATERING]");
    } else {
        display.printf("Pump: OFF [IDLE]");
    }

    // Push the buffer to the physical display.
    display.display();
}


// =============================================================
//  HTTP POST — Send Telemetry to the Server
// =============================================================
// Returns: true if the server responded with "water": true,
//          false otherwise (including on connection failure).
bool postTelemetry(float moisture, float temperature) {
    // Build the full URL: e.g., "http://192.168.1.100:8000/ingest/moisture"
    // We POST to /ingest/moisture because that's the endpoint the
    // FastAPI server exposes (see server/main.py line 334).
    String url = String(SERVER_BASE_URL) + "/ingest/moisture";

    Serial.printf("  [HTTP] POSTing to %s\n", url.c_str());

    // ── Build JSON payload ──────────────────────
    // ArduinoJson uses a "document" pattern:
    //   1. Create a JsonDocument (statically sized buffer).
    //   2. Add key-value pairs.
    //   3. Serialize to a String.
    //
    // The server expects: {"value": 42.5}
    // We also include temperature for future use — the server
    // currently ignores extra fields, but this makes the payload
    // ready for when you add a /ingest/telemetry endpoint.
    JsonDocument doc;
    doc["value"] = moisture;

    String jsonPayload;
    serializeJson(doc, jsonPayload);

    Serial.printf("  [HTTP] Payload: %s\n", jsonPayload.c_str());

    // ── Send the HTTP POST ──────────────────────
    HTTPClient http;
    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    // Timeout: 10 seconds. If the server doesn't respond in time,
    // we don't want the ESP32 hanging forever.
    http.setTimeout(10000);

    int httpResponseCode = http.POST(jsonPayload);

    bool shouldWater = false;

    if (httpResponseCode > 0) {
        // HTTP request succeeded — parse the response.
        String responseBody = http.getString();
        Serial.printf("  [HTTP] Response (%d): %s\n", httpResponseCode, responseBody.c_str());

        if (httpResponseCode == 200) {
            // Parse the JSON response.
            // The server returns: {"water": true/false, "received_value": 42.5}
            JsonDocument responseDoc;
            DeserializationError error = deserializeJson(responseDoc, responseBody);

            if (error) {
                Serial.printf("  [HTTP] JSON parse error: %s\n", error.c_str());
            } else {
                // Extract the "water" boolean — this is the server's decision.
                shouldWater = responseDoc["water"].as<bool>();
                Serial.printf("  [HTTP] Server says water=%s\n",
                              shouldWater ? "TRUE" : "FALSE");
            }
        } else {
            Serial.printf("  [HTTP] Server returned error code: %d\n", httpResponseCode);
        }
    } else {
        // HTTP request failed entirely (network error, timeout, etc.)
        Serial.printf("  [HTTP] POST failed, error: %s\n",
                      http.errorToString(httpResponseCode).c_str());
        Serial.println("         → Is the server running? Check IP and port.");
    }

    // Always free resources when done.
    http.end();

    return shouldWater;
}


// =============================================================
//  PUMP CONTROL
// =============================================================

void activatePump() {
    // Turn the relay ON → this closes the circuit → pump runs.
    // Most relay modules are ACTIVE LOW: writing LOW to the signal
    // pin energizes the relay coil and closes the switch.
    //
    // SAFETY: We record the start time so the auto-shutoff in loop()
    // can enforce the maximum pump duration. This prevents the relay
    // from staying on indefinitely if something goes wrong.
    Serial.println("  [PUMP] >>> Activating pump! Watering started.");
    digitalWrite(RELAY_PIN, LOW);  // LOW = relay ON (active-low)
    pumpIsRunning = true;
    pumpStartTime = millis();

    // Update the OLED immediately to show pump status change.
    updateOLED(WiFi.status() == WL_CONNECTED, currentMoisture, currentTemperature, true);
}

void deactivatePump() {
    // Turn the relay OFF → circuit opens → pump stops.
    Serial.println("  [PUMP] <<< Deactivating pump. Watering complete.");
    digitalWrite(RELAY_PIN, HIGH);  // HIGH = relay OFF (active-low)
    pumpIsRunning = false;

    // Update the OLED immediately to show pump status change.
    updateOLED(WiFi.status() == WL_CONNECTED, currentMoisture, currentTemperature, false);
}


// =============================================================
//  IMAGE CAPTURE & POST (ESP32-CAM — FUTURE IMPLEMENTATION)
// =============================================================
// This function will capture a JPEG from the ESP32-CAM's OV2640
// camera module and POST it to /ingest/image for disease detection.
//
// TODO: Uncomment and implement when switching to ESP32-CAM.
//       The ESP32-CAM uses a different board configuration and
//       pin mapping. You will need to:
//       1. Select "AI Thinker ESP32-CAM" as your board in Arduino IDE.
//       2. Add #include "esp_camera.h" at the top of this file.
//       3. Configure the camera pins for the AI Thinker module.
//       4. Initialize the camera in setup() with a JPEG config.
//
// The server's /ingest/image endpoint expects:
//   - HTTP POST with multipart/form-data
//   - Field name: "file"
//   - Content-Type: image/jpeg
//   - Body: raw JPEG bytes (must start with 0xFF 0xD8 magic bytes)

void captureAndPostImage() {
    /*
    // ── Step 1: Capture a frame from the camera ──────────
    // The esp_camera library provides fb (frame buffer) objects.
    // Each fb contains the raw JPEG bytes from the OV2640 sensor.
    //
    // camera_fb_t* fb = esp_camera_fb_get();
    // if (!fb) {
    //     Serial.println("[CAM] ERROR: Camera capture failed!");
    //     return;
    // }
    // Serial.printf("[CAM] Captured image: %d bytes\n", fb->len);

    // ── Step 2: Build the multipart/form-data request ────
    // Multipart encoding wraps the binary JPEG in a text envelope
    // with boundaries, so the server can extract the file.
    //
    // Think of it like putting a photo in an envelope with a label
    // that says "this is a JPEG named plant.jpg" — the server reads
    // the label and extracts the photo.
    //
    // String url = String(SERVER_BASE_URL) + "/ingest/image";
    // HTTPClient http;
    // http.begin(url);
    //
    // // Generate a unique boundary string for multipart encoding.
    // String boundary = "----ESP32Boundary" + String(millis());
    // http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
    //
    // // Build the multipart body:
    // //   --boundary
    // //   Content-Disposition: form-data; name="file"; filename="plant.jpg"
    // //   Content-Type: image/jpeg
    // //   <blank line>
    // //   <JPEG bytes>
    // //   --boundary--
    //
    // String head = "--" + boundary + "\r\n"
    //               "Content-Disposition: form-data; name=\"file\"; filename=\"plant.jpg\"\r\n"
    //               "Content-Type: image/jpeg\r\n\r\n";
    // String tail = "\r\n--" + boundary + "--\r\n";
    //
    // // Calculate total content length for the Content-Length header.
    // uint32_t totalLen = head.length() + fb->len + tail.length();
    //
    // // Use the streaming API to send the multipart body in chunks.
    // // This avoids loading the entire payload into a single String
    // // (which could exceed the ESP32's available RAM for large images).
    // WiFiClient* stream = http.getStreamPtr();
    //
    // http.addHeader("Content-Length", String(totalLen));
    //
    // // We need to use a lower-level approach for streaming:
    // WiFiClient client;
    // if (client.connect(SERVER_IP, SERVER_PORT)) {
    //     client.println("POST /ingest/image HTTP/1.1");
    //     client.printf("Host: %s:%d\r\n", SERVER_IP, SERVER_PORT);
    //     client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    //     client.printf("Content-Length: %u\r\n", totalLen);
    //     client.println("Connection: close");
    //     client.println();  // End of headers
    //
    //     // Send the multipart header
    //     client.print(head);
    //
    //     // Send the JPEG bytes in chunks (4096 bytes at a time)
    //     const size_t CHUNK_SIZE = 4096;
    //     uint8_t* fbBuf = fb->buf;
    //     size_t fbLen = fb->len;
    //     for (size_t offset = 0; offset < fbLen; offset += CHUNK_SIZE) {
    //         size_t chunkLen = min(CHUNK_SIZE, fbLen - offset);
    //         client.write(fbBuf + offset, chunkLen);
    //     }
    //
    //     // Send the multipart tail
    //     client.print(tail);
    //
    //     // Read the server's response
    //     Serial.println("[CAM] Image sent. Waiting for response...");
    //     while (client.connected()) {
    //         String line = client.readStringUntil('\n');
    //         if (line == "\r") break;  // End of HTTP headers
    //     }
    //     String response = client.readString();
    //     Serial.printf("[CAM] Server response: %s\n", response.c_str());
    //
    //     client.stop();
    // } else {
    //     Serial.println("[CAM] ERROR: Could not connect to server.");
    // }

    // ── Step 3: Release the frame buffer ─────────────────
    // CRITICAL: Always return the buffer to the camera driver.
    // The ESP32-CAM has limited frame buffers (usually 1–2).
    // Forgetting to release causes a memory leak and the camera
    // will stop capturing after a few frames.
    //
    // esp_camera_fb_return(fb);
    */

    // Placeholder — this function does nothing on a standard ESP32.
    Serial.println("[CAM] Image capture not available — standard ESP32 has no camera.");
    Serial.println("      Switch to ESP32-CAM and uncomment this function.");
}
