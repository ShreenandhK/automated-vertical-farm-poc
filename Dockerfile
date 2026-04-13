# ============================================================
# Dockerfile — Plant Health Monitoring System (POC)
# Base: Python 3.11 slim for a small image footprint.
# Strategy: Copy requirements first, install deps, THEN copy
# source code. This leverages Docker's layer caching — if your
# code changes but dependencies don't, the slow pip install
# step is skipped on rebuild.
# ============================================================

FROM python:3.11-slim

# Set the working directory inside the container.
# Every subsequent COPY, RUN, and CMD will execute relative to /app.
WORKDIR /app

# --- Dependency Layer (cached unless requirements.txt changes) ---
# Copy ONLY the requirements file first so Docker can cache this
# expensive layer independently of source code changes.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir: prevents pip from storing downloaded wheel files,
#   keeping the image smaller (no point caching inside a container).
# --no-compile: skips generating .pyc files during install (they'll
#   be created at runtime anyway), shaving a few MB off the image.
RUN pip install --no-cache-dir --no-compile -r requirements.txt

# --- Application Layer (rebuilt on every code change) ---
# Copy the rest of the project into the container.
# Files listed in .dockerignore (secrets, datasets, .git, etc.)
# are automatically excluded — that's why we configured it first.
COPY . .

# Create the saved_images/ directory inside the container.
# Why? The app mounts this as a volume at runtime, but if Docker
# starts the container BEFORE the volume mount is ready (or if
# someone runs the image without docker-compose), the app would
# crash trying to write to a non-existent directory. This is a
# safety net — the volume mount will overlay this empty dir.
RUN mkdir -p saved_images

# Tell Docker (and humans reading this) that the app listens on 8000.
# Note: EXPOSE is documentation only — it doesn't actually publish
# the port. The actual mapping happens in docker-compose.yml.
EXPOSE 8000

# Launch the FastAPI application via Uvicorn.
# --host 0.0.0.0: binds to ALL network interfaces inside the
#   container, making it reachable from the host machine.
#   (localhost would only be visible inside the container itself.)
# --port 8000: matches our EXPOSE declaration above.
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
