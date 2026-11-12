# ==============================================================================
# Fraud Detection System (FDS) - Production Dockerfile
# Author: Alpha
# Multi-stage build for minimal attack surface and fast startup
# ==============================================================================

# Stage 1: Build & Dependency Wheel Cache
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compile dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# Stage 2: Production Runtime
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime PostgreSQL client library & curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Create non-root application user for container hardening
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy application source code
COPY --chown=appuser:appgroup . .

# Ensure directory permissions for runtime artifacts
RUN mkdir -p /app/ml/artifacts /app/data/raw && \
    chown -R appuser:appgroup /app

USER appuser

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    PORT=5000

EXPOSE 5000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Production Gunicorn WSGI entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
