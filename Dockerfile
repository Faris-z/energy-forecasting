# ─────────────────────────────────────────────────────────────────────────────
#  Energy Forecasting — Dockerfile
#  Multi-stage build: slim final image with all ML dependencies.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a dedicated prefix so we can copy just the packages
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="energy-forecasting" \
      org.opencontainers.image.description="Deep learning time series forecasting pipeline" \
      org.opencontainers.image.version="1.0.0"

# Runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project source code
COPY src/           ./src/
COPY config/        ./config/
COPY tests/         ./tests/
COPY train.py       .
COPY prepare_data.py .
COPY Makefile       .
COPY requirements.txt .

# Create output directories
RUN mkdir -p \
    data/raw \
    data/processed \
    models \
    reports/figures \
    logs

# Non-root user for security
RUN useradd --create-home --shell /bin/bash forecaster && \
    chown -R forecaster:forecaster /app

USER forecaster

# Expose MLflow UI port
EXPOSE 5000

# Default: run full pipeline
CMD ["make", "all"]
