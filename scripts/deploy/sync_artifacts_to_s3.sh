#!/usr/bin/env bash
# Upload local data + model artifacts to the project's S3 buckets, mirroring the
# architecture's storage zones.
#
# Usage:
#   ./scripts/deploy/sync_artifacts_to_s3.sh <artifacts_bucket> [curated_bucket] [raw_bucket] [region]
set -euo pipefail

ARTIFACTS_BUCKET="${1:?artifacts bucket name required}"
CURATED_BUCKET="${2:-}"
RAW_BUCKET="${3:-}"
REGION="${4:-us-east-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f models/tft_model.ckpt ]]; then
  echo "==> Uploading model checkpoint -> s3://${ARTIFACTS_BUCKET}/models/"
  aws s3 cp models/tft_model.ckpt "s3://${ARTIFACTS_BUCKET}/models/tft_model.ckpt" --region "$REGION"
fi

if [[ -n "$CURATED_BUCKET" && -f data/processed/clean.csv ]]; then
  echo "==> Uploading clean.csv -> s3://${CURATED_BUCKET}/"
  aws s3 cp data/processed/clean.csv "s3://${CURATED_BUCKET}/clean.csv" --region "$REGION"
fi

if [[ -n "$RAW_BUCKET" && -d data/raw ]]; then
  echo "==> Syncing raw CSVs -> s3://${RAW_BUCKET}/"
  aws s3 sync data/raw "s3://${RAW_BUCKET}/" --exclude ".gitkeep" --region "$REGION"
fi

echo "Done."
