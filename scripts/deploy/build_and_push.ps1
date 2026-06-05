<#
.SYNOPSIS
  Build the DS-02 Docker image and push it to Amazon ECR.

.DESCRIPTION
  Logs in to ECR, builds the image from the repo root (which bakes in the
  trained checkpoint + clean.csv that must already exist on disk), and pushes
  it under the given tag. App Runner (auto_deployments_enabled = true) then
  redeploys automatically.

.EXAMPLE
  .\scripts\deploy\build_and_push.ps1 -Region us-east-1 -Repo ds02-demand-forecasting -Tag latest
#>
param(
  [string]$Region = "us-east-1",
  [string]$Repo   = "ds02-demand-forecasting",
  [string]$Tag    = "latest"
)
$ErrorActionPreference = "Stop"

# Run from the repo root so the Docker build context is correct.
$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."
Push-Location $RepoRoot
try {
  if (-not (Test-Path "models\tft_model.ckpt") -or -not (Test-Path "data\processed\clean.csv")) {
    throw "Missing artifacts. Run 'python run_all.py --synthetic' (or pull them from S3) before building."
  }

  $AccountId = (aws sts get-caller-identity --query Account --output text)
  $Registry  = "$AccountId.dkr.ecr.$Region.amazonaws.com"
  $Image     = "${Registry}/${Repo}:${Tag}"

  Write-Host "==> Logging in to ECR ($Registry)"
  aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry

  Write-Host "==> Building $Image"
  docker build -t $Image .

  Write-Host "==> Pushing $Image"
  docker push $Image

  Write-Host "Done. Pushed $Image"
  Write-Host "App Runner will auto-deploy the new image if the service already exists."
}
finally {
  Pop-Location
}
