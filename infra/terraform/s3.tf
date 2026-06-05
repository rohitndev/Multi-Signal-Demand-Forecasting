data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
}

# force_destroy lets `terraform destroy` remove the (versioned) buckets even when
# they still contain objects — convenient for a demo/teardown.

# Raw / landing zone — where ingestion drops the unprocessed Rossmann + signal
# CSVs (architecture: "Amazon S3 Raw / Landing Zone").
resource "aws_s3_bucket" "raw" {
  bucket        = "${var.project_name}-raw-${local.account_id}"
  force_destroy = true
}

# Curated / processed zone — clean.csv and the model-ready feature matrix
# (architecture: "Amazon S3 Processed / Curated Zone").
resource "aws_s3_bucket" "curated" {
  bucket        = "${var.project_name}-curated-${local.account_id}"
  force_destroy = true
}

# Model artifacts + MLflow store (the TFT checkpoint, drift reports, etc.).
resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project_name}-artifacts-${local.account_id}"
  force_destroy = true
}

locals {
  buckets = {
    raw       = aws_s3_bucket.raw.id
    curated   = aws_s3_bucket.curated.id
    artifacts = aws_s3_bucket.artifacts.id
  }
}

# --- Hardening applied uniformly to all three buckets ---
resource "aws_s3_bucket_versioning" "this" {
  for_each = local.buckets
  bucket   = each.value

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = local.buckets
  bucket   = each.value

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = local.buckets
  bucket   = each.value

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
