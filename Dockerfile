# ==============================================================================
# Base Image: Python 3.12 Slim
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and force unbuffered standard output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    CANSAT_HARDWARE=off

# ==============================================================================
# Essential Runtime Libraries for OpenCV & ML (Headless Graphics & OpenMP)
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    -o Acquire::Retries=3 \
    -o Acquire::http::Timeout=60 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Security & Permissions: Non-Root User (UID 1000 for Hugging Face Spaces)
# ==============================================================================
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ==============================================================================
# Dependency Layer Caching
# ==============================================================================
COPY --chown=user:user requirements.txt .

# Install Python dependencies under the non-root user
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# Application Source Code
# ==============================================================================
COPY --chown=user:user . .

# Expose FastAPI application port
EXPOSE 8000

# Run Uvicorn binding to 0.0.0.0 (Supports dynamic PORT env override)
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
