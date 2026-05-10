/*
 * ============================================================================
 *  plant_monitor.ino — ESP32-CAM Firmware for SIP Vertical Farm Monitor
 * ============================================================================
 *
 *  WHAT THIS DOES:
 *    1. Connects to your local WiFi network.
 *    2. Reads soil moisture (analog) and temperature (DHT22) on a timer.
 *    3. POSTs telemetry as JSON to the Python/FastAPI server.
 *    4. Parses the server's response and controls a relay-driven water pump.
 *    5. Captures JPEG images via the OV3660 camera and POSTs them to the
 *       server for AI-based plant disease detection.
 *
 *  HARDWARE (AI Thinker ESP32-CAM form factor):
 *    - ESP32-CAM with OV3660 camera (3 MP sensor — successor to the OV2640)
 *    - Analog Soil Moisture Sensor  → GPIO 13  ⚠ See PIN NOTE below
 *    - DHT22 Temperature Sensor     → GPIO 4   (with 10kΩ pull-up)
 *    - 5V Relay Module              → GPIO 14  ⚠ See PIN NOTE below
 *    - Submersible Water Pump       → wired through the relay
 *    - Programming: USB-to-UART adapter on GPIO 1 (TX) / GPIO 3 (RX)
 *
 *  ⚠ PIN NOTE — Why pins changed from the original ESP32 DevKit wiring:
 *    The OV2640 camera on the ESP32-CAM uses almost every GPIO:
 *      - GPIO 34 (was SOIL sensor) → now Y8 (camera data bus)
 *      - GPIO 26 (was RELAY)       → now SIOD (camera I2C SDA)
 *    These two pins are UNAVAILABLE for external use when the camera is active.
 *    New assignments use GPIO 13 (moisture) and GPIO 14 (relay), which are
 *    tied to the SD card slot on the PCB but are free if you do not use SD.
 *
 *    Additional caveat: GPIO 13 is on ADC2. The ESP32 shares ADC2 with its
 *    WiFi radio, which causes brief interference during active WiFi traffic.
 *    Averaging 10+ samples (already done below) largely compensates for this.
 *    If you need higher accuracy, consider a digital threshold moisture sensor
 *    (e.g. FC-28 in digital mode) — digital reads are not affected by ADC2.
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
 *    ESP32 responsive — it can handle WiFi events and pump auto-shutoff
 *    even between sensor readings and image captures.
 *
 *  AUTHOR:  SIP Project — Phase 7 Hardware Integration (ESP32-CAM)
 *  LICENSE: MIT
 * ============================================================================
 */

// ─────────────────────────────────────────────────────────────
// LIBRARY INCLUDES
// ─────────────────────────────────────────────────────────────

// WiFi and HTTP — built into the ESP32 Arduino core.
#include <WiFi.h>
#include <HTTPClient.h>

// ESP32 camera driver — part of the ESP32 Arduino core (espressif32 platform).
// No extra library install needed; it ships with the board support package.
// The driver auto-detects the sensor model over SCCB (I2C), so the same
// API works for OV2640, OV3660, OV5640, etc. — only the init parameters
// (XCLK frequency, frame size limits) need to be tuned per sensor.
#include "esp_camera.h"

// JSON serialization/deserialization — ArduinoJson v7.
#include <ArduinoJson.h>

// DHT temperature/humidity sensor.
// (Install via Library Manager: "DHT sensor library" by Adafruit)
#include <DHT.h>


// ─────────────────────────────────────────────────────────────
// AI THINKER ESP32-CAM — CAMERA PIN MAPPING
// ─────────────────────────────────────────────────────────────
// These are the fixed hardware connections on the AI Thinker PCB.
// Do NOT change these — they are soldered traces, not configurable.
// They are identical whether the soldered sensor is OV2640 or OV3660,
// because the pin map describes the BOARD, not the sensor module.
// If you have a different ESP32-CAM variant (e.g. M5Camera, TTGO,
// ESP32-S3-CAM, Freenove), look up its specific pin map and replace
// these values — the sensor data lines often map to different GPIOs.

// Official AI Thinker ESP32-CAM pin map per Espressif's reference design.
// Source: github.com/espressif/esp32-camera/blob/master/examples/CameraWebServer
// Verified working with the same hardware that successfully captured a JPEG
// in the standalone hardware test sketch.
#define PWDN_GPIO_NUM     32   // Camera power-down pin
#define RESET_GPIO_NUM    -1   // No dedicated reset pin on this module
#define XCLK_GPIO_NUM      0   // External clock output to the sensor
#define SIOD_GPIO_NUM     26   // SCCB (I2C) data — camera config bus
#define SIOC_GPIO_NUM     27   // SCCB (I2C) clock — camera config bus
#define Y9_GPIO_NUM       35   // Pixel data bus D7 (MSB)
#define Y8_GPIO_NUM       34   // Pixel data bus D6
#define Y7_GPIO_NUM       39   // Pixel data bus D5
#define Y6_GPIO_NUM       36   // Pixel data bus D4
#define Y5_GPIO_NUM       21   // Pixel data bus D3
#define Y4_GPIO_NUM       19   // Pixel data bus D2
#define Y3_GPIO_NUM       18   // Pixel data bus D1
#define Y2_GPIO_NUM        5   // Pixel data bus D0 (LSB)
#define VSYNC_GPIO_NUM    25   // Vertical sync — marks start of a new frame
#define HREF_GPIO_NUM     23   // Horizontal reference — marks start of a line
#define PCLK_GPIO_NUM     22   // Pixel clock — clocks each pixel out


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
// IMPORTANT: The ESP32-CAM and server MUST be on the same local network.
const char* SERVER_BASE_URL = "http://YOUR_SERVER_IP:8000";

// How often to read sensors and send telemetry (in milliseconds).
const unsigned long TELEMETRY_INTERVAL_MS = 10000;  // 10 seconds

// How often to capture and POST a camera image (in milliseconds).
// Image POSTs are heavier than telemetry (JPEG ~20–80 KB vs <200 bytes),
// so we do them less frequently to avoid saturating the network or server.
const unsigned long IMAGE_INTERVAL_MS = 30000;  // 30 seconds

// How long to keep the pump running when the server says "water: true".
// This matches the WATERING_DURATION_SECONDS=5 in your .env file.
const unsigned long PUMP_DURATION_MS = 5000;  // 5 seconds

// WiFi reconnection attempt interval — if WiFi drops, retry every 5 seconds.
const unsigned long WIFI_RETRY_INTERVAL_MS = 5000;


// ─────────────────────────────────────────────────────────────
// PIN DEFINITIONS
// ─────────────────────────────────────────────────────────────
// ⚠ These differ from the original ESP32 DevKit wiring.
// See the hardware section at the top of this file for an explanation.

// Soil Moisture Sensor — analog input.
// GPIO 13 is ADC2 ch4. Reads may be slightly noisier during WiFi activity.
// We compensate by averaging 10 samples (see readSoilMoisture()).
const int SOIL_MOISTURE_PIN = 13;

// DHT22 Temperature & Humidity Sensor — digital pin.
// GPIO 4 is also the flash LED pin on the AI Thinker. It will NOT trigger
// the flash here because we never configure it as OUTPUT. DHT uses it as
// a bidirectional open-drain pin, which does not light the LED.
const int DHT_PIN  = 4;
const int DHT_TYPE = DHT22;  // Use DHT11 if you have that model instead

// Relay Module — digital output.
// GPIO 14 is safe as a general-purpose output if the SD card is not used.
// ⚠ BOOT CAUTION: GPIO 12 on ESP32-CAM must be LOW at boot (it's a
//   flash voltage strap). GPIO 14 has no such restriction and is preferred.
const int RELAY_PIN = 14;


// ─────────────────────────────────────────────────────────────
// SENSOR CALIBRATION
// ─────────────────────────────────────────────────────────────
// HOW TO CALIBRATE:
//   1. Read the sensor in DRY AIR     → that's your DRY_VALUE (high number).
//   2. Submerge the sensor in WATER   → that's your WET_VALUE (low number).
//   Note: Capacitive sensors read HIGH when dry, LOW when wet (inverse).
//   Your specific sensor may differ — adjust these after testing.
const int SOIL_DRY_VALUE = 3500;  // Raw ADC when sensor is in dry air
const int SOIL_WET_VALUE = 1500;  // Raw ADC when sensor is submerged in water


// ─────────────────────────────────────────────────────────────
// GLOBAL OBJECTS
// ─────────────────────────────────────────────────────────────

// DHT sensor object — handles the one-wire protocol internally.
DHT dht(DHT_PIN, DHT_TYPE);


// ─────────────────────────────────────────────────────────────
// STATE VARIABLES (non-blocking timing)
// ─────────────────────────────────────────────────────────────
// Instead of delay(), we track "when did we last do X?" and check
// if enough time has passed. This is the millis() pattern.
//
// Analogy: Imagine you're cooking. Instead of standing at the stove
// watching the timer (delay), you set a mental note and go do other
// things, glancing at the clock periodically (millis check).

unsigned long lastTelemetryTime = 0;  // When we last read sensors + POSTed
unsigned long lastImageTime     = 0;  // When we last captured + POSTed an image
unsigned long pumpStartTime     = 0;  // When the pump was turned ON
unsigned long lastWiFiRetryTime = 0;  // When we last tried to reconnect WiFi
bool          pumpIsRunning     = false;  // Is the pump currently active?

// Latest sensor readings — stored globally so Serial logs can reference
// them from helper functions (activatePump, deactivatePump, etc.).
float currentMoisture    = 0.0;
float currentTemperature = 0.0;


// ─────────────────────────────────────────────────────────────
// FUNCTION DECLARATIONS (forward declarations for readability)
// ─────────────────────────────────────────────────────────────
bool  initCamera();
void  connectToWiFi();
float readSoilMoisture();
float readTemperature();
bool  postTelemetry(float moisture, float temperature);
void  captureAndPostImage();
void  activatePump();
void  deactivatePump();


// =============================================================
//  SETUP — Runs once when the ESP32-CAM powers on or resets
// =============================================================
void setup() {
    // ── Serial Monitor ──────────────────────────
    Serial.begin(115200);
    while (!Serial) { ; }  // Wait for serial port to be ready
    Serial.println();
    Serial.println("============================================");
    Serial.println("  SIP Plant Monitor — ESP32-CAM Firmware v2.0");
    Serial.println("============================================");

    // ── Pin Modes ───────────────────────────────
    pinMode(SOIL_MOISTURE_PIN, INPUT);

    // Relay pin: OUTPUT, starting HIGH (relay OFF for active-low modules).
    // Explicitly setting this BEFORE camera init ensures the relay does
    // not accidentally activate during the camera initialization sequence.
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, HIGH);
    Serial.println("[INIT] Relay pin set to OUTPUT, pump OFF.");

    // ── DHT22 Sensor ────────────────────────────
    dht.begin();
    Serial.println("[INIT] DHT22 sensor initialized.");

    // ── Camera ──────────────────────────────────
    // Camera init must happen before WiFi is started.
    // The camera driver configures the SCCB (I2C) bus to talk to the OV2640,
    // which uses GPIO 26 and 27. Initializing WiFi first can conflict with
    // peripherals that share the I2C bus.
    if (!initCamera()) {
        // Camera failure is logged but does not halt the firmware.
        // Telemetry and watering still work — the system degrades gracefully.
        Serial.println("[ERROR] Camera init failed. Image capture will be skipped.");
    }

    // ── WiFi Connection ─────────────────────────
    connectToWiFi();

    Serial.println("[INIT] Setup complete. Entering main loop.");
    Serial.println("--------------------------------------------");
}


// =============================================================
//  LOOP — Runs continuously after setup()
// =============================================================
void loop() {
    unsigned long now = millis();

    // ── WiFi Health Check ───────────────────────
    if (WiFi.status() != WL_CONNECTED) {
        if (now - lastWiFiRetryTime >= WIFI_RETRY_INTERVAL_MS) {
            Serial.println("[WIFI] Connection lost. Attempting reconnect...");
            connectToWiFi();
            lastWiFiRetryTime = now;
        }
    }

    // ── Telemetry Cycle ─────────────────────────
    if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
        lastTelemetryTime = now;

        currentMoisture    = readSoilMoisture();
        currentTemperature = readTemperature();

        Serial.println("──── Telemetry Cycle ────");
        Serial.printf("  Moisture:    %.1f%%\n", currentMoisture);
        Serial.printf("  Temperature: %.1f°C\n", currentTemperature);

        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("  [SKIP] No WiFi — telemetry not sent.");
        }
        // SAFETY GATE: never trigger the pump on a sentinel/invalid reading.
        // currentMoisture == -1.0 means every ADC2 read this cycle timed out
        // (WiFi was holding the bus). Sending -1 to the server would make it
        // think the soil is bone-dry and request watering on EVERY cycle —
        // which, with a real pump connected, would flood the plants.
        // We'd rather skip a cycle than over-water on garbage data.
        else if (currentMoisture < 0) {
            Serial.println("  [SKIP] Moisture invalid — telemetry suppressed.");
            Serial.println("         (Watering decisions need real ADC data.)");
        }
        else {
            bool shouldWater = postTelemetry(currentMoisture, currentTemperature);
            if (shouldWater && !pumpIsRunning) {
                activatePump();
            }
        }

        Serial.println("─────────────────────────");
    }

    // ── Image Capture Cycle ─────────────────────
    // Run on a slower cadence than telemetry — image POSTs are heavy.
    if (now - lastImageTime >= IMAGE_INTERVAL_MS) {
        lastImageTime = now;

        if (WiFi.status() == WL_CONNECTED) {
            captureAndPostImage();
        } else {
            Serial.println("[CAM] No WiFi — image capture skipped.");
        }
    }

    // ── Pump Auto-Shutoff ───────────────────────
    // Safety: enforce a maximum pump run time regardless of server state.
    // Analogy: Like a kitchen timer that turns off the oven automatically.
    if (pumpIsRunning && (now - pumpStartTime >= PUMP_DURATION_MS)) {
        deactivatePump();
    }

    // Yield to the ESP32 system/WiFi task scheduler.
    yield();
}


// =============================================================
//  CAMERA INITIALIZATION
// =============================================================
bool initCamera() {
    // Build the camera configuration struct using the AI Thinker pin macros
    // defined at the top of this file.
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    // Assign camera data bus and control pins from our macros above.
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;

    // XCLK frequency — the master clock the ESP32 generates for the sensor.
    //
    // 20 MHz is the standard, well-tested frequency for both OV2640 and
    // OV3660 on the AI Thinker ESP32-CAM. We tried 10 MHz earlier as a
    // "safer" value, but that's actually TOO SLOW for some OV3660 units —
    // the sensor's SCCB (I2C) state machine fails to respond to register
    // reads during init, leaving the driver with a NULL sensor pointer
    // that crashes with LoadProhibited at EXCVADDR 0x0F.
    //
    // If you see stripe artifacts in captured images, lower this to
    // 16 MHz (16000000). Don't go below 10 MHz on OV3660.
    config.xclk_freq_hz = 20000000;

    // Output format: JPEG. Both OV2640 and OV3660 have an on-chip JPEG
    // encoder, saving us the CPU cost of software compression.
    config.pixel_format = PIXFORMAT_JPEG;

    // Frame size and quality depend on whether the module has PSRAM.
    // The AI Thinker ESP32-CAM has 4 MB PSRAM soldered on the PCB.
    // PSRAM acts as overflow RAM for large frame buffers — without it,
    // the only RAM available is the ESP32's 320 KB SRAM, which limits
    // you to tiny (QVGA 320×240) frames.
    //
    // psramFound() returns true if PSRAM was detected and is usable.
    // GRAB MODE — how the driver behaves when frame buffers fill up.
    //
    // CAMERA_GRAB_WHEN_EMPTY (default): driver pauses capture when buffers
    //     are full and prints "cam_hal: FB-OVF" warnings until we read one.
    //     Bad fit for our use case — we capture only every 30 seconds, but
    //     the camera DMA keeps streaming frames continuously in between.
    //
    // CAMERA_GRAB_LATEST: driver overwrites the oldest buffer with new
    //     frames as they arrive. When we finally call esp_camera_fb_get(),
    //     we get the MOST RECENT frame and no FB-OVF spam. This is exactly
    //     what we want for periodic capture: we don't care about frames
    //     between calls, we just want the freshest one when we ask.
    config.grab_mode = CAMERA_GRAB_LATEST;

    if (psramFound()) {
        // SVGA (800×600) gives good disease-detection detail while keeping
        // file size manageable over WiFi (~30–60 KB per JPEG).
        //
        // OV3660 NOTE: This sensor can do up to QXGA (2048×1536) since it's
        // a 3 MP part, but the disease-detection model resizes inputs to
        // ~224×224 anyway, so larger frames just waste bandwidth and PSRAM.
        // Stick with SVGA unless you have a specific reason to go bigger.
        config.frame_size   = FRAMESIZE_SVGA;
        config.jpeg_quality = 12;   // 0 = best quality, 63 = worst. 10–15 is a good range.
        config.fb_count     = 2;    // Two frame buffers: one being captured while the other is being sent.
        config.fb_location  = CAMERA_FB_IN_PSRAM;  // Store buffers in PSRAM, not SRAM.
    } else {
        // No PSRAM: fall back to QVGA with one buffer to fit in SRAM.
        Serial.println("[CAM] WARNING: PSRAM not found. Using low-res fallback (QVGA).");
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAM] esp_camera_init() failed with error 0x%x\n", err);
        Serial.println("      → Check that the camera ribbon cable is seated correctly.");
        Serial.println("      → Verify power: the camera needs a stable 3.3 V supply.");
        Serial.println("      → If error is 0x103 (ESP_ERR_NOT_FOUND), the SCCB bus");
        Serial.println("        couldn't reach the sensor — try lowering xclk_freq_hz.");
        return false;
    }

    // Query the driver to confirm which sensor was detected.
    // This is informational — useful when swapping OV2640 ↔ OV3660 boards.
    sensor_t* s = esp_camera_sensor_get();
    if (s != nullptr) {
        Serial.printf("[INIT] Camera initialized. Sensor PID: 0x%02X (",
                      s->id.PID);
        switch (s->id.PID) {
            case OV2640_PID: Serial.print("OV2640"); break;
            case OV3660_PID: Serial.print("OV3660"); break;
            case OV5640_PID: Serial.print("OV5640"); break;
            default:         Serial.print("unknown"); break;
        }
        Serial.println(")");

        // OV3660 tuning: the default register set on this sensor tends to
        // produce slightly washed-out, low-contrast images straight out of
        // the box. These tweaks bring it closer to natural color.
        // Skip these on OV2640 — its defaults are already well-tuned.
        if (s->id.PID == OV3660_PID) {
            s->set_brightness(s, 1);     // Slight brightness lift (-2 to 2)
            s->set_saturation(s, -2);    // Default OV3660 over-saturates greens
            s->set_contrast(s, 0);       // Leave neutral
            Serial.println("[CAM] Applied OV3660 default-look corrections.");
        }
    }

    // ── Sensor warm-up ─────────────────────────────────────────────
    // The OV3660's auto-exposure and auto-white-balance algorithms need
    // several frames to settle before producing a usable image. If you
    // call esp_camera_fb_get() too soon after init, it returns NULL or
    // a black/garbage frame.
    //
    // Analogy: like a digital camera that beeps and flashes "PROCESSING"
    // for a moment after you turn it on — we wait for that moment, then
    // throw away the first few frames as the AE/AWB locks in.
    Serial.println("[CAM] Warming up sensor (discarding first frames)...");
    delay(500);
    for (int i = 0; i < 3; i++) {
        camera_fb_t* warmup = esp_camera_fb_get();
        if (warmup) {
            esp_camera_fb_return(warmup);
        }
        delay(100);
    }
    Serial.println("[CAM] Warm-up complete.");

    return true;
}


// =============================================================
//  WiFi CONNECTION
// =============================================================
void connectToWiFi() {
    Serial.printf("[WIFI] Connecting to \"%s\"", WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    // Wait up to 15 seconds during initial boot.
    // After boot, reconnection is handled non-blocking in loop().
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
    // Capacitive sensors output HIGHER voltage when DRY and LOWER when WET,
    // so the ADC mapping is INVERTED: high ADC value → dry → low moisture %.
    //
    // ⚠ ESP32-CAM ADC2/WiFi CONFLICT ⚠
    // ──────────────────────────────────────────────────────────────────
    // Every analog-capable GPIO broken out on the AI Thinker ESP32-CAM
    // (GPIO 2, 12, 13, 14) sits on ADC2. The ADC2 hardware block is
    // SHARED with the WiFi radio — when WiFi is active, analogRead()
    // frequently returns 0 with an ESP_ERR_TIMEOUT. We can't move the
    // sensor to ADC1 because no ADC1 pins are exposed on this board.
    //
    // STRATEGY:
    //   1. Take many samples; FILTER OUT the zeros (failed reads).
    //   2. Average only the SUCCESSFUL reads.
    //   3. If every read failed, return the last KNOWN-GOOD reading.
    //
    // The static `lastGoodPercentage` survives between calls — it acts
    // like a memory of "the last time the sensor actually worked."
    // Think of it as a doctor saying "I couldn't get a clean pulse this
    // time — using the patient's last reading until the next checkup."

    static float lastGoodPercentage = 50.0;  // Sensible default at first boot
    static bool  hasGoodReading     = false;

    const int NUM_SAMPLES = 20;   // More samples → more chance of success
    long sum = 0;
    int  validCount = 0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        int raw = analogRead(SOIL_MOISTURE_PIN);
        // Skip readings of 0 — these are ADC2 timeout failures, not real
        // "completely wet" readings (a real soaked sensor reads ~1500).
        if (raw > 0) {
            sum += raw;
            validCount++;
        }
        delayMicroseconds(200);  // Slightly longer settle time helps ADC2
    }

    // No valid readings at all this cycle? Reuse the last good value.
    if (validCount == 0) {
        Serial.printf("  [SOIL] All %d reads timed out (ADC2 busy). Using last value: %.1f%%\n",
                      NUM_SAMPLES, hasGoodReading ? lastGoodPercentage : -1.0f);
        return hasGoodReading ? lastGoodPercentage : -1.0f;
    }

    int rawValue = sum / validCount;

    // Map raw ADC (DRY_VALUE→0%, WET_VALUE→100%), clamped to the calibration range.
    int constrained = constrain(rawValue, SOIL_WET_VALUE, SOIL_DRY_VALUE);
    float percentage = map(constrained, SOIL_DRY_VALUE, SOIL_WET_VALUE, 0, 100);
    percentage = constrain(percentage, 0.0f, 100.0f);

    // Cache this successful reading for future cycles where ADC2 is busy.
    lastGoodPercentage = percentage;
    hasGoodReading     = true;

    Serial.printf("  [SOIL] Raw ADC: %d (%d/%d valid) → Moisture: %.1f%%\n",
                  rawValue, validCount, NUM_SAMPLES, percentage);
    return percentage;
}


// =============================================================
//  TEMPERATURE READING
// =============================================================
float readTemperature() {
    // The DHT22 returns NaN if the read fails (loose wiring, missing pull-up,
    // or reading too fast — the DHT22 needs ~2 seconds between reads).
    float tempC = dht.readTemperature();

    if (isnan(tempC)) {
        Serial.println("  [DHT] ERROR: Failed to read temperature!");
        Serial.println("         → Check wiring and 10kΩ pull-up resistor on GPIO 4.");
        return -999.0;  // Sentinel value so the server knows the reading is invalid
    }

    Serial.printf("  [DHT] Temperature: %.1f°C\n", tempC);
    return tempC;
}


// =============================================================
//  HTTP POST — Send Telemetry to the Server
// =============================================================
// Returns: true if the server responded with "water": true,
//          false otherwise (including on connection failure).
bool postTelemetry(float moisture, float temperature) {
    String url = String(SERVER_BASE_URL) + "/ingest/moisture";
    Serial.printf("  [HTTP] POSTing to %s\n", url.c_str());

    // Build JSON payload: {"value": 42.5}
    // We include temperature as an extra field for future server-side use.
    // The current /ingest/moisture endpoint ignores extra fields gracefully.
    JsonDocument doc;
    doc["value"] = moisture;

    String jsonPayload;
    serializeJson(doc, jsonPayload);
    Serial.printf("  [HTTP] Payload: %s\n", jsonPayload.c_str());

    HTTPClient http;
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000);  // 10-second timeout to avoid hanging the loop

    int httpResponseCode = http.POST(jsonPayload);
    bool shouldWater = false;

    if (httpResponseCode > 0) {
        String responseBody = http.getString();
        Serial.printf("  [HTTP] Response (%d): %s\n", httpResponseCode, responseBody.c_str());

        if (httpResponseCode == 200) {
            JsonDocument responseDoc;
            DeserializationError error = deserializeJson(responseDoc, responseBody);

            if (error) {
                Serial.printf("  [HTTP] JSON parse error: %s\n", error.c_str());
            } else {
                shouldWater = responseDoc["water"].as<bool>();
                Serial.printf("  [HTTP] Server says water=%s\n",
                              shouldWater ? "TRUE" : "FALSE");
            }
        } else {
            Serial.printf("  [HTTP] Server returned error code: %d\n", httpResponseCode);
        }
    } else {
        Serial.printf("  [HTTP] POST failed: %s\n",
                      http.errorToString(httpResponseCode).c_str());
        Serial.println("         → Is the server running? Check IP and port.");
    }

    http.end();
    return shouldWater;
}


// =============================================================
//  IMAGE CAPTURE & POST
// =============================================================
void captureAndPostImage() {
    // ── Step 1: Capture a JPEG frame from the OV2640 ─────────────
    // esp_camera_fb_get() asks the camera driver for a filled frame buffer.
    // The OV2640 compresses the image to JPEG in its own hardware DSP,
    // so `fb->buf` already contains a valid JPEG when this returns.
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[CAM] ERROR: Frame capture failed! Skipping image POST.");
        Serial.println("      → This can happen if PSRAM is full or the sensor is stalled.");
        return;
    }
    Serial.printf("[CAM] Captured JPEG: %d bytes (frame %dx%d)\n",
                  fb->len, fb->width, fb->height);

    // ── Step 2: Stream multipart/form-data to the server ──────────
    // Multipart encoding wraps the binary JPEG in a text "envelope" with
    // boundary markers, so the server can extract the file from the body.
    //
    // Analogy: putting a photo in a labelled envelope — the server reads
    // the label ("this is a JPEG named plant.jpg") and pulls out the photo.
    //
    // We use a raw WiFiClient instead of HTTPClient here because HTTPClient
    // would buffer the entire body in RAM before sending. For a JPEG that
    // can be 20–80 KB, that risks a heap overflow. The raw client lets us
    // stream the JPEG bytes in small chunks, keeping RAM usage flat.

    String boundary = "----ESP32Boundary" + String(millis());
    String head = "--" + boundary + "\r\n"
                  "Content-Disposition: form-data; name=\"file\"; filename=\"plant.jpg\"\r\n"
                  "Content-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--" + boundary + "--\r\n";

    uint32_t totalLen = head.length() + fb->len + tail.length();

    // Parse host and port from SERVER_BASE_URL for the raw TCP connection.
    // Example: "http://192.168.1.100:8000" → host="192.168.1.100", port=8000
    String url  = String(SERVER_BASE_URL);
    String host = url.substring(7);                // Strip "http://"
    int    port = 8000;                             // Default
    int    colonIdx = host.indexOf(':');
    if (colonIdx != -1) {
        port = host.substring(colonIdx + 1).toInt();
        host = host.substring(0, colonIdx);
    }

    WiFiClient client;
    if (!client.connect(host.c_str(), port)) {
        Serial.printf("[CAM] ERROR: Could not connect to %s:%d\n", host.c_str(), port);
        Serial.println("      → Is the FastAPI server running?");
        esp_camera_fb_return(fb);  // Always return the buffer even on failure
        return;
    }

    // Send HTTP request line and headers.
    client.printf("POST /ingest/image HTTP/1.1\r\n");
    client.printf("Host: %s:%d\r\n", host.c_str(), port);
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    client.printf("Content-Length: %u\r\n", totalLen);
    client.printf("Connection: close\r\n");
    client.printf("\r\n");  // Blank line marks end of headers

    // Send the multipart header (text part).
    client.print(head);

    // Send the JPEG bytes in 4 KB chunks to avoid RAM pressure.
    // Writing 80 KB in one call would require an 80 KB contiguous heap block —
    // chunking keeps each write small and predictable.
    const size_t CHUNK_SIZE = 4096;
    uint8_t* buf = fb->buf;
    size_t   len = fb->len;
    for (size_t offset = 0; offset < len; offset += CHUNK_SIZE) {
        size_t chunkLen = min(CHUNK_SIZE, len - offset);
        client.write(buf + offset, chunkLen);
    }

    // Send the multipart tail (closing boundary).
    client.print(tail);

    // ── Step 3: Read the server's response ───────────────────────
    Serial.println("[CAM] Image sent. Waiting for server response...");

    // Wait up to 30 seconds for the server to reply.
    //
    // Why 30s and not 5s? When the local TFLite confidence is low, the
    // server falls back to Claude Vision or Gemini — those round trips
    // can take 5–15 seconds depending on cloud latency. A 5-second cap
    // would cut off the response mid-stream, which is exactly the empty
    // "[CAM] Server response:" line we saw in the earlier output.
    const unsigned long RESPONSE_TIMEOUT_MS = 30000;
    unsigned long timeout = millis() + RESPONSE_TIMEOUT_MS;
    while (client.available() == 0 && millis() < timeout) {
        delay(10);
    }

    if (client.available() == 0) {
        Serial.println("[CAM] WARNING: server response timed out after 30s.");
        Serial.println("      → Cloud AI fallback may be slow; check server logs.");
        client.stop();
        esp_camera_fb_return(fb);
        return;
    }

    // Skip the HTTP response headers (read until the blank line).
    // setTimeout() configures how long readStringUntil() waits for more
    // data when the buffer empties mid-read — important for chunked
    // responses streamed from FastAPI's async handlers.
    client.setTimeout(RESPONSE_TIMEOUT_MS);
    while (client.connected() || client.available()) {
        String line = client.readStringUntil('\n');
        if (line == "\r" || line.length() == 0) break;  // Blank line = end of headers
    }

    // Read and log the JSON response body.
    String response = client.readString();
    Serial.printf("[CAM] Server response: %s\n", response.c_str());

    client.stop();

    // ── Step 4: Return the frame buffer — CRITICAL ────────────────
    // The ESP32-CAM driver has a fixed pool of frame buffers (1–2 slots).
    // Forgetting to call fb_return() permanently consumes that slot —
    // the next call to esp_camera_fb_get() will then return NULL.
    // Always return the buffer, even when an error occurred above.
    esp_camera_fb_return(fb);
}


// =============================================================
//  PUMP CONTROL
// =============================================================

void activatePump() {
    // Turn relay ON (active-low: writing LOW energizes the coil).
    // Record the start time so loop() can enforce auto-shutoff.
    Serial.println("  [PUMP] >>> Activating pump! Watering started.");
    digitalWrite(RELAY_PIN, LOW);
    pumpIsRunning = true;
    pumpStartTime = millis();
}

void deactivatePump() {
    // Turn relay OFF — circuit opens, pump stops.
    Serial.println("  [PUMP] <<< Deactivating pump. Watering complete.");
    digitalWrite(RELAY_PIN, HIGH);
    pumpIsRunning = false;
}
