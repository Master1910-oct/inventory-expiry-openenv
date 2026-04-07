# ── Inventory & Expiry Management OpenEnv ─────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="inventory-env"
LABEL org.opencontainers.image.title="Inventory & Expiry Management OpenEnv"
LABEL org.opencontainers.image.description="Real-world inventory management RL environment"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY server/requirements.txt /tmp/requirements.txt
COPY requirements-inference.txt /tmp/requirements-inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/requirements-inference.txt

# Copy source
COPY . /app

# Expose port (HF Spaces default)
ENV PORT=7860
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]