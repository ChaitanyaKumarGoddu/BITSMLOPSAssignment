# syntax=docker/dockerfile:1
# ---- Task 6: containerise the model-serving API ----
# Multi-stage build keeps the final image small: wheels are built once in the
# builder stage, then only the runtime venv + app code land in the final image.

# ---------- Stage 1: build dependencies ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
# Build all wheels up front so the runtime stage installs offline & fast.
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Run as a non-root user (production hygiene).
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Install runtime deps from the pre-built wheels.
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code and the trained model artifact.
COPY src/ ./src/
COPY models/ ./models/

USER appuser
EXPOSE 8000

# Container-native healthcheck hitting the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
