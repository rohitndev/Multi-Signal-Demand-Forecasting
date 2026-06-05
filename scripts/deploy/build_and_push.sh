#!/usr/bin/env bash
# Build the DS-02 Docker image and push it to Amazon ECR.
#
# Usage:
#   ./scripts/deploy/build_and_push.sh [region] [repo] [tag]
# Example:
#   ./scripts/deploy/build_and_push.sh us-east-1 ds02-demand-forecasting latest
set -euo pipefail

REGION="${1:-us-east-1}"
REPO="${2:-ds02-demand-forecasting}"
TAG="${3:-latest}"

# Run from the repo root so the Docker build context is correct.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f models/tft_model.ckpt || ! -f data/processed/clean.csv ]]; then
  echo "Missing artifacts. Run 'python run_all.py --synthetic' (or pull them from S3) before building." >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

echo "==> Logging in to ECR (${REGISTRY})"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

echo "==> Building ${IMAGE}"
docker build -t "$IMAGE" .

echo "==> Pushing ${IMAGE}"
docker push "$IMAGE"

echo "Done. Pushed ${IMAGE}"
echo "App Runner will auto-deploy the new image if the service already exists."
