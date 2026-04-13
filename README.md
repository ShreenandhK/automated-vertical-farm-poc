# IoT + AI Plant Health Monitoring & Automated Watering System

> **A Proof of Concept for AI-Assisted Vertical Farming**
>
> Built with FastAPI · TFLite (ai-edge-litert) · Google Gemini Vision · WebSockets · SQLite · Docker

---

## Table of Contents

1. [Project Overview & AI Workflow](#1-project-overview--ai-workflow)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Prerequisites & Initial Setup](#3-prerequisites--initial-setup)
4. [Running the Project Locally](#4-running-the-project-locally-development-mode)
5. [Running the Hardware Simulator](#5-running-the-hardware-simulator)
6. [Docker Containerization](#6-docker-containerization-production-mode)
7. [Accessing the Live Dashboard](#7-accessing-the-live-dashboard)

---

## 1. Project Overview & AI Workflow

### What Is This?

This system is a **Proof of Concept (POC)** for an automated vertical farming assistant. It does two things simultaneously:

1. **Monitors soil moisture** and automatically triggers a watering pump when levels drop too low.
2. **Analyzes plant images** in real time using a two-stage AI pipeline to detect diseases early — before they spread.

The philosophy is **Human-in-the-Loop**: the system automates the *routine* (watering, continuous monitoring) but always notifies a human when something *critical* happens (disease detected). It never takes irreversible action on its own. A farmer sees the alert on the live dashboard and decides what to do next.

---

### The Hybrid AI Pipeline — Plain English

Think of the system like a hospital triage process:

```
[ESP32-CAM or Simulator]
        |
        | sends a JPEG photo via HTTP POST
        v
┌─────────────────────────────────────────────────────────┐
│  Step 1 — "The Bouncer" (server/main.py)                │
│  FastAPI receives the image and checks it's a valid     │
│  JPEG (magic bytes check). Rejects anything else.       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────┐
│  Step 2 — "The Instinct" (server/inference.py)          │
│  A lightweight, 64×64 int8-quantized TFLite model       │
│  runs entirely on your machine — no internet needed.    │
│  It classifies the image across 33 PlantVillage         │
│  categories and aggregates them into two buckets:       │
│  "Healthy" or "Diseased", with a confidence score.      │
└─────────────────────────┬───────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
     confidence >= 0.75        confidence < 0.75
      (certain enough)          (not sure enough)
              │                       │
              v                       v
    ✅ Accept local result    ┌───────────────────────────┐
    Fast, free, offline.      │ Step 3 — "The Specialist" │
                              │  (server/cloud_fallback.py)│
                              │                           │
                              │  Image is sent to a cloud │
                              │  AI vision model for a    │
                              │  definitive diagnosis:    │
                              │                           │
                              │  • Google Gemini (primary)│
                              │  • Anthropic Claude       │
                              │    (cascade fallback)     │
                              │                           │
                              │  Returns structured JSON: │
                              │  disease name, confidence,│
                              │  recommended action.      │
                              └───────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────┐
│  Step 4 — Log & Alert                                   │
│  Every result is saved to SQLite (data_logger.py).      │
│  If disease is detected, a WebSocket alert fires to     │
│  all connected dashboard clients (alert_manager.py).    │
└─────────────────────────────────────────────────────────┘
```

**Why two stages?** The local model is fast (milliseconds, no cost), but it was trained on a limited dataset and can be uncertain. The cloud model is slower and costs API credits, but it's far more capable. By only escalating *uncertain* cases, we get the best of both worlds: speed and accuracy, used intelligently.

---

## 2. Project Folder Structure

```
SIP-project/
│
├── server/                     ← The "Brain" — all backend Python code lives here
│   ├── main.py                 ← Traffic controller: FastAPI app, all HTTP routes, startup logic
│   ├── inference.py            ← The "Instinct": loads and runs the local TFLite model
│   ├── cloud_fallback.py       ← The "Specialist": routes to Gemini or Claude when local is uncertain
│   ├── data_logger.py          ← Librarian: saves every analysis result to the SQLite database
│   ├── alert_manager.py        ← Town crier: broadcasts WebSocket alerts to the live dashboard
│   ├── moisture_controller.py  ← Watering logic: triggers pump when soil moisture is too low
│   │
│   ├── models/                 ← Where the AI brain lives
│   │   ├── plant_disease_model.tflite  ← The local TFLite model file (gitignored if large)
│   │   └── labels.txt                 ← 33 PlantVillage class names, one per line
│   │
│   └── static/                 ← The "Face" — live dashboard served directly by FastAPI
│       ├── index.html          ← Dashboard structure and layout
│       ├── app.js              ← WebSocket client, chart updates, image display logic
│       └── style.css           ← Dashboard styling
│
├── simulator/                  ← Fake hardware for testing without real sensors
│   └── mock_device.py          ← Acts as a fake ESP32-CAM + soil moisture sensor
│
├── tests/                      ← Automated tests
│   ├── test_local_ai.py        ← Tests for the local inference pipeline
│   └── test_cloud_api.py       ← Tests for the cloud fallback pipeline
│
├── data/                       ← Sample images for the simulator to send
│   └── plantvillage_samples/   ← PlantVillage dataset samples, organised by species/condition
│       ├── Pepper,_bell___Bacterial_spot/   (15 diseased pepper images)
│       ├── Pepper,_bell___healthy/          (15 healthy pepper images)
│       ├── Potato___Early_blight/           (15 diseased potato images)
│       ├── Potato___healthy/                (15 healthy potato images)
│       ├── Strawberry___Leaf_scorch/        (15 diseased strawberry images)
│       └── Strawberry___healthy/            (15 healthy strawberry images)
│
├── saved_images/               ← Every uploaded image is stored here (Docker volume mount)
│
├── .env                        ← 🔐 Your secrets and config — NEVER commit this file
├── .gitignore                  ← Tells Git to ignore .env, the database, saved images, etc.
├── requirements.txt            ← Pinned Python package versions
├── Dockerfile                  ← Recipe for building the Docker container image
├── docker-compose.yml          ← Orchestrates the container (ports, volumes, env vars)
└── CLAUDE.md                   ← Project memory and coding standards (for AI-assisted dev)
```

---

## 3. Prerequisites & Initial Setup

### What You Need Installed

| Tool | Why You Need It | Download |
|---|---|---|
| **Python 3.11** | Runs the backend server | [python.org](https://www.python.org/downloads/) |
| **Docker Desktop** | Runs the containerised production version | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Google AI Studio API Key** | Powers the cloud AI fallback (Gemini) | [aistudio.google.com](https://aistudio.google.com/apikey) |

> **Note:** For local development, you only need Python 3.11. Docker is only required for the containerised production mode (Section 6).

---

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd SIP-project
```

---

### Step 2 — Create the `.env` File

The `.env` file is where all your secrets and configuration values live. It is intentionally **excluded from Git** (listed in `.gitignore`) so your API keys are never accidentally committed to version control.

Create a file named `.env` in the root of the project:

```bash
# On Windows (PowerShell or Command Prompt)
copy .env.example .env   # if an example file exists, or create manually
```

Paste the following into `.env` and fill in your values:

```dotenv
# ─────────────────────────────────────────────
# AI Configuration
# Only ONE cloud key is required. The system
# auto-detects which provider to use.
# ─────────────────────────────────────────────

# Leave blank if you don't have an Anthropic key
ANTHROPIC_API_KEY=

# Get yours free at: https://aistudio.google.com/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Confidence gate: if the local model scores BELOW this,
# the image is escalated to the cloud AI.
# Range: 0.0 (escalate everything) to 1.0 (never escalate)
CONFIDENCE_THRESHOLD=0.75

# ─────────────────────────────────────────────
# Local AI Model Paths
# ─────────────────────────────────────────────
TFLITE_MODEL_PATH=server/models/plant_disease_model.tflite
TFLITE_LABELS_PATH=server/models/labels.txt

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
DATABASE_URL=sqlite:///./server/plant_monitor.db

# ─────────────────────────────────────────────
# Watering Controller
# ─────────────────────────────────────────────
MOISTURE_LOW_THRESHOLD=30    # Below this % → trigger watering
MOISTURE_HIGH_THRESHOLD=70   # Above this % → stop watering
WATERING_DURATION_SECONDS=5  # How many seconds to run the pump

# ─────────────────────────────────────────────
# Image Storage
# ─────────────────────────────────────────────
IMAGE_SAVE_DIR=saved_images/

# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
```

> **Why `HOST=0.0.0.0`?** This tells the server to listen on *all* network interfaces, not just your own computer. This is essential for Docker (so the container is reachable from the outside) and for accessing the dashboard from your phone on the same WiFi network.

---

### Step 3 — Set Up the Sample Image Dataset

The simulator needs images to send to the server. The `data/plantvillage_samples/` directory should already contain the sample images organised by species and condition. Verify this structure exists:

```
data/
└── plantvillage_samples/
    ├── Pepper,_bell___Bacterial_spot/   ← ~15 diseased pepper images
    ├── Pepper,_bell___healthy/          ← ~15 healthy pepper images
    ├── Potato___Early_blight/           ← ~15 diseased potato images
    ├── Potato___healthy/                ← ~15 healthy potato images
    ├── Strawberry___Leaf_scorch/        ← ~15 diseased strawberry images
    └── Strawberry___healthy/            ← ~15 healthy strawberry images
```

If images are missing, download a sample of the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and organise them into the structure above.

---

## 4. Running the Project Locally (Development Mode)

This mode runs the server directly on your machine — no Docker needed. It's the fastest way to develop and test.

### Step 1 — Set Up the Python Environment

**Using Conda (recommended):**

```bash
# Create a fresh environment with Python 3.11
conda create -n sip-project python=3.11 -y

# Activate it
conda activate sip-project

# Install all required packages (exact versions are pinned)
pip install -r requirements.txt
```

**Using venv (alternative):**

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2 — Start the Server

From the **root of the project** (not inside the `server/` folder), run:

```bash
uvicorn server.main:app --reload
```

The `--reload` flag means the server **automatically restarts** whenever you save a change to a Python file — incredibly useful during development.

---

### Step 3 — Verify It Started Successfully

You should see output like this in your terminal:

```
INFO:     Will watch for changes in these directories: ['/path/to/SIP-project']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [XXXX] using WatchFiles
INFO:root:TFLite model loaded successfully. Input shape: (1, 64, 64, 3), dtype: int8
INFO:root:Loaded 33 labels from server/models/labels.txt
INFO:root:Database tables created/verified.
INFO:     Started server process [XXXX]
INFO:     Waiting for connections.
INFO:     Application startup complete.
```

**What each line means:**
- `Uvicorn running on http://0.0.0.0:8000` — The server is live and accepting connections.
- `TFLite model loaded successfully` — The local AI model is in memory, ready to analyse images.
- `Loaded 33 labels` — The model knows the names of all 33 plant conditions it can detect.
- `Database tables created/verified` — SQLite is ready to log results.
- `Application startup complete` — Everything is healthy. Open your browser!

If you see any **red error messages** at startup, the most common causes are:
- Missing `.env` file or a typo in a path.
- Missing model file at `server/models/plant_disease_model.tflite`.
- A package not installed — re-run `pip install -r requirements.txt`.

---

## 5. Running the Hardware Simulator

The simulator (`simulator/mock_device.py`) is your **fake ESP32-CAM and soil moisture sensor**. Since we don't have physical hardware during development, the simulator reads real images from your `data/plantvillage_samples/` directory and sends them to the server via HTTP — exactly as the real device would.

Think of it like a fire drill: everything runs exactly as it would in production, but no real hardware is needed.

> **Prerequisite:** The server must already be running (Section 4) before you launch the simulator. Open a **second terminal** for the simulator commands below.

---

### Simulator Modes

#### Test with Healthy Plant Images

```bash
python -m simulator.mock_device --mode healthy --interval 3
```

**What this does:** Every **3 seconds**, picks a random image from a *healthy* plant subfolder and POSTs it to the server. Also sends a simulated moisture reading. Use this to verify the "no disease" path works correctly and that the dashboard stays calm.

---

#### Test with Diseased Plant Images

```bash
python -m simulator.mock_device --mode diseased --interval 10
```

**What this does:** Every **10 seconds**, picks a random image from a *diseased* plant subfolder and POSTs it. You should see the server escalate uncertain cases to the cloud AI and fire WebSocket alerts to the dashboard. Watch the dashboard light up!

---

#### Simulate a Real Mixed Environment

```bash
python -m simulator.mock_device --mode real --interval 30
```

**What this does:** Every **30 seconds**, picks a completely **random** image from *any* subfolder — healthy or diseased. This simulates a real-world scenario where the system doesn't know in advance what it will see. The best mode for realistic end-to-end testing.

---

### What You'll See in the Simulator Terminal

```
[mock_device] 2025-04-14 12:01:03 | Sending image: Potato___Early_blight/image_001.jpg
[mock_device] 2025-04-14 12:01:03 | Moisture reading: 24% (below threshold — watering triggered)
[mock_device] 2025-04-14 12:01:04 | Server response: {"status": "disease_detected", "source": "cloud", "confidence": 0.91}
```

---

## 6. Docker Containerization (Production Mode)

### What Is Docker?

Imagine your project is a complex machine. Getting it to run on your computer requires installing Python, all the right packages, setting up paths, etc. Docker packages the **entire machine — Python, packages, code, and all** — into a single "shipping container" (an *image*). Anyone with Docker Desktop can run that container and get an identical, working environment in seconds.

This is why Docker is standard practice for deploying applications: it eliminates "it works on my machine" problems entirely.

---

### Build and Run the Container

From the project root (where `docker-compose.yml` lives):

```bash
docker compose up --build
```

**What this does, step by step:**
1. **`--build`** reads the `Dockerfile` and builds a fresh container image with your code and all dependencies baked in.
2. Docker Compose starts the container, maps port `8000` on your machine to port `8000` inside the container.
3. It loads your `.env` file and passes those values as environment variables into the container.
4. The server starts inside the container, exactly as it does in local dev mode.

You'll see the same startup logs as in Section 4. The server is accessible at `http://localhost:8000`.

---

### Run in the Background (Detached Mode)

If you don't want the terminal occupied, run:

```bash
docker compose up -d
```

The `-d` flag means **detached** — the container runs as a background process. To check its status:

```bash
docker compose ps       # Is it running?
docker compose logs -f  # Stream the live logs
docker compose down     # Stop and remove the container
```

---

### Why Volume Mounts Matter

Look at this section in `docker-compose.yml`:

```yaml
volumes:
  - ./plant_monitor.db:/app/plant_monitor.db
  - ./saved_images:/app/saved_images
  - ./server/models:/app/server/models
```

Containers are **ephemeral by default** — if you stop and remove a container, everything inside it is gone. Volume mounts solve this by linking folders *inside* the container to folders *on your real machine*:

| Volume | What It Protects |
|---|---|
| `plant_monitor.db` | Your entire analysis history (SQLite database) survives container restarts. |
| `saved_images/` | Every uploaded plant photo is stored on your real disk, not lost when the container stops. |
| `server/models/` | You can swap the TFLite model file without rebuilding the entire container. |

---

## 7. Accessing the Live Dashboard

### On Your Computer

Once the server is running (locally or via Docker), open your browser and go to:

```
http://localhost:8000/static/index.html
```

---

### On Your Smartphone (Same WiFi Network)

You can access the dashboard from your phone or tablet — no app required, just a browser. Here's how:

**Step 1:** Find your computer's local IP address.

```bash
# On Windows (in PowerShell or Command Prompt):
ipconfig

# Look for "IPv4 Address" under your active network adapter.
# It will look something like: 192.168.1.105
```

**Step 2:** On your phone's browser, navigate to:

```
http://192.168.1.105:8000/static/index.html
```

Replace `192.168.1.105` with your actual IPv4 address. As long as both devices are on the same WiFi network, you'll see the live dashboard.

> **Why does this work?** Because the server binds to `HOST=0.0.0.0`, it listens on *all* network interfaces — including your WiFi adapter. Your phone can reach it using your computer's local IP.

---

### What You'll See on the Dashboard

The dashboard connects to the server via a **WebSocket** — a persistent, two-way connection that lets the server *push* updates to your browser instantly (no page refresh needed).

| Element | What It Shows |
|---|---|
| **Disease Alert Panel** | Fires a real-time notification when a disease is detected, including the plant name, disease type, AI confidence score, and the source (local model or cloud AI). |
| **Latest Plant Image** | Displays the most recently analysed plant photo. |
| **Soil Moisture Gauge** | Live percentage reading from the moisture sensor (or simulator). Turns red when below the `MOISTURE_LOW_THRESHOLD`. |
| **Analysis History** | A log of all previous analysis results pulled from the SQLite database. |

---

## Quick Reference — Common Commands

```bash
# ── Development Mode ────────────────────────────────────────────
conda activate sip-project            # Activate Python environment
uvicorn server.main:app --reload      # Start the server with auto-reload

# ── Simulator ───────────────────────────────────────────────────
python -m simulator.mock_device --mode healthy   --interval 3    # Send healthy images
python -m simulator.mock_device --mode diseased  --interval 10   # Send diseased images
python -m simulator.mock_device --mode real      --interval 30   # Send random images

# ── Docker / Production Mode ─────────────────────────────────────
docker compose up --build             # Build and start container (foreground)
docker compose up -d                  # Start container in background
docker compose logs -f                # Stream live logs from container
docker compose down                   # Stop and remove container

# ── Dashboard URLs ───────────────────────────────────────────────
# Local:    http://localhost:8000/static/index.html
# Network:  http://<your-IPv4>:8000/static/index.html
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on startup | Missing dependency | Run `pip install -r requirements.txt` |
| `FileNotFoundError` for `.tflite` model | Model file missing | Ensure `server/models/plant_disease_model.tflite` exists |
| Cloud fallback returns an error | Invalid or missing API key | Check `GEMINI_API_KEY` in your `.env` file |
| Dashboard shows no data | WebSocket not connecting | Make sure server is running and refresh the page |
| Docker container exits immediately | Startup error inside container | Run `docker compose logs` to see the error message |
| Simulator says "Connection refused" | Server not running | Start the server first, then run the simulator |

---

## Acknowledgements & Data Sources

This project stands on the shoulders of open research and the generosity of the Kaggle and academic communities.

---

### AI Model — Shrunken EfficientNetLite (TFLite)

The local TFLite model used in this project (`server/models/plant_disease_model.tflite`) is sourced from:

> **Timothy Lovett** — *Plant Disease Shrunken TFLite Quantization*
> Kaggle Notebook & Dataset: [timothylovett/plant-disease-shrunken-tflite-quantization](https://www.kaggle.com/code/timothylovett/plant-disease-shrunken-tflite-quantization/notebook)
> Dataset: [timothylovett/shrunken-efficientnetlite-plants-disease](https://www.kaggle.com/datasets/timothylovett/shrunken-efficientnetlite-plants-disease)

The model is an **int8-quantized EfficientNetLite** architecture, trained on the PlantVillage dataset and compressed ("shrunken") to run efficiently on edge devices. It accepts **64×64 RGB images** and classifies them across **33 PlantVillage categories**, which our inference pipeline aggregates into a binary "Healthy" vs. "Diseased" result.

> **If you need the model file:** Download `plant_disease_model.tflite` and `labels.txt` from the Kaggle dataset above and place them in `server/models/`.

---

### Plant Image Dataset — PlantVillage

The sample images in `data/plantvillage_samples/` are drawn from the **PlantVillage dataset**:

> **Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016)**
> *Using Deep Learning for Image-Based Plant Disease Detection.*
> *Frontiers in Plant Science, 7, 1419.*
> GitHub Repository: [spmohanty/plantvillage-dataset](https://github.com/spmohanty/plantvillage-dataset)

The PlantVillage dataset contains **54,309 images** across 38 classes of healthy and diseased plant leaves, covering 14 crop species. It is the benchmark dataset for agricultural disease detection research and is what the TFLite model above was trained on.

> **To populate `data/plantvillage_samples/`:** Download a subset of images from the GitHub repository above or from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), and organise them into the species/condition subfolder structure described in Section 3.

---

*Built as part of a Social Internship Project (SIP) — a proof of concept for intelligent, human-assisted vertical farming.*
