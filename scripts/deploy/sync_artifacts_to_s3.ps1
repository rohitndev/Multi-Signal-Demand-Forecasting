<#
.SYNOPSIS
  Upload local data + model artifacts to the project's S3 buckets.

.DESCRIPTION
  Mirrors the architecture's storage zones: raw CSVs -> raw bucket,
  clean.csv -> curated bucket, the TFT checkpoint -> artifacts bucket.
  Pass the bucket names printed by `terraform output`.

.EXAMPLE
  .\scripts\deploy\sync_artifacts_to_s3.ps1 `
    -ArtifactsBucket multi-signal-demand-forecasting-artifacts-123456789012 `
    -CuratedBucket  multi-signal-demand-forecasting-curated-123456789012 `
    -RawBucket      multi-signal-demand-forecasting-raw-123456789012
#>
param(
  [Parameter(Mandatory = $true)][string]$ArtifactsBucket,
  [string]$CuratedBucket = "",
  [string]$RawBucket = "",
  [string]$Region = "us-east-1"
)
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $RepoRoot
try {
  if (Test-Path "models\tft_model.ckpt") {
    Write-Host "==> Uploading model checkpoint -> s3://$ArtifactsBucket/models/"
    aws s3 cp "models\tft_model.ckpt" "s3://$ArtifactsBucket/models/tft_model.ckpt" --region $Region
  }

  if ($CuratedBucket -and (Test-Path "data\processed\clean.csv")) {
    Write-Host "==> Uploading clean.csv -> s3://$CuratedBucket/"
    aws s3 cp "data\processed\clean.csv" "s3://$CuratedBucket/clean.csv" --region $Region
  }

  if ($RawBucket -and (Test-Path "data\raw")) {
    Write-Host "==> Syncing raw CSVs -> s3://$RawBucket/"
    aws s3 sync "data\raw" "s3://$RawBucket/" --exclude ".gitkeep" --region $Region
  }

  Write-Host "Done."
}
finally {
  Pop-Location
}
