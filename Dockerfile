# syntax=docker/dockerfile:1
###############################################################################
# DS-02 Multi-Signal Demand Forecasting — API container image
#
# Serves the FastAPI app (src.api:app) on port 8000. The trained TFT checkpoint
# and the cleaned dataset are baked in, so the container is self-contained and
# ready for AWS App Runner (image pulled from Amazon ECR).
#
# Build (from the repo root, after producing the artifacts with run_all.py):
#     docker build -t ds02-demand-forecasting .
# Run locally:
#     docker run -p 8000:8000 ds02-demand-forecasting
###############################################################################
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keep caches inside writable, non-root-owned temp dirs.
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl

# libgomp1 is required at runtime by torch / statsmodels; curl is handy for
# container-level health checks / debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch from the dedicated index first so we don't pull the
# multi-gigabyte CUDA build, then install everything else (torch is already
# satisfied, so the pinned `torch==` line in requirements.txt is skipped).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && grep -ivE '^torch==' requirements.txt > /tmp/requirements.no-torch.txt \
    && pip install -r /tmp/requirements.no-torch.txt

# Application code.
COPY src ./src
COPY scripts ./scripts
COPY run_all.py ./run_all.py

# Artifacts the API serves. These are produced by `python run_all.py` (or pulled
# from S3 in CI) and are git-ignored, so they must exist on disk at build time.
COPY models/tft_model.ckpt ./models/tft_model.ckpt
COPY data/processed/clean.csv ./data/processed/clean.csv

# Run as a non-root user.
RUN useradd --create-home appuser \
    && mkdir -p /tmp/hf /tmp/mpl \
    && chown -R appuser:appuser /app /tmp/hf /tmp/mpl
USER appuser

# App Runner routes traffic to this port (keep in sync with infra/terraform).
EXPOSE 8000

# Bind 0.0.0.0 so the App Runner proxy (and `docker run -p`) can reach uvicorn.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
