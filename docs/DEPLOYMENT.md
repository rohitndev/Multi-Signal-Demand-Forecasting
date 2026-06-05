# AWS Deployment Guide — DS-02 Demand Forecasting

This guide deploys the FastAPI service to **AWS App Runner**, pulling a Docker
image from **Amazon ECR**, with **S3** storage zones, **IAM** roles, and
**CloudWatch** alarms — all provisioned by **Terraform**. It matches the
*"Infrastructure (AWS)"* and *"Deployment"* lanes of the architecture diagram.

> The application code is unchanged — it still runs locally exactly as documented
> in the main [README](../README.md). This is **deployment config only**.

```text
                         GitHub Actions (CI/CD)
                                  │ docker build & push
                                  ▼
  data/processed/clean.csv  ┌───────────┐      ┌──────────────────┐
  models/tft_model.ckpt ───▶│ Docker img │ ───▶ │   Amazon ECR     │
        (baked in)          └───────────┘      └────────┬─────────┘
                                                         │ pull (ECR access role)
                                                         ▼
   S3: raw / curated / artifacts  ◀──IAM──▶   AWS App Runner ──▶ HTTPS endpoint
                                                         │
                                                         ▼
                                                  Amazon CloudWatch
                                                  (logs, metrics, alarms)
```

---

## 1. Prerequisites

| Tool | Purpose |
|------|---------|
| [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) | Auth + ECR login + S3 sync |
| [Docker](https://docs.docker.com/get-docker/) | Build the serving image |
| [Terraform ≥ 1.5](https://developer.hashicorp.com/terraform/downloads) | Provision AWS infra |
| An AWS account + credentials | `aws configure` (an admin/deploy IAM user or SSO) |

Verify access:

```bash
aws sts get-caller-identity
```

---

## 2. One-time: produce the serving artifacts

The image bakes in the trained checkpoint and the cleaned dataset. Generate them
once (synthetic data is fine for a demo; use the real Kaggle data for production):

```bash
python run_all.py --synthetic --epochs 5
# -> models/tft_model.ckpt
# -> data/processed/clean.csv
```

---

## 3. Provision infrastructure with Terraform

App Runner can only start once an image exists in ECR, so apply in **two phases**.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # optional: edit region/sizing
terraform init

# Phase 1 — create just the registry (+ its dependencies) so we can push an image.
terraform apply -target=aws_ecr_repository.app
```

Note the repository URL:

```bash
terraform output -raw ecr_repository_url
# e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com/ds02-demand-forecasting
```

---

## 4. Build & push the image

From the **repo root**:

```bash
# Linux / macOS
./scripts/deploy/build_and_push.sh us-east-1 ds02-demand-forecasting latest
```

```powershell
# Windows PowerShell
.\scripts\deploy\build_and_push.ps1 -Region us-east-1 -Repo ds02-demand-forecasting -Tag latest
```

These scripts log in to ECR, run `docker build .`, and push the `latest` tag.

---

## 5. Create the rest of the stack (App Runner, S3, IAM, CloudWatch)

```bash
cd infra/terraform
terraform apply
```

When it finishes:

```bash
terraform output -raw service_url
# https://xxxxx.us-east-1.awsapprunner.com
```

Smoke-test the live endpoint:

```bash
URL=$(terraform output -raw service_url)
curl "$URL/"
curl -X POST "$URL/forecast" -H "Content-Type: application/json" \
  -d '{"store_id": 1, "forecast_days": 28}'
```

Interactive docs are at `"$URL/docs"`.

---

## 6. (Optional) Populate the S3 storage zones

The running container serves the baked-in artifacts, but you can also mirror data
into the S3 zones shown in the architecture (raw → curated → artifacts):

```bash
cd infra/terraform
./../../scripts/deploy/sync_artifacts_to_s3.sh \
  "$(terraform output -raw artifacts_bucket)" \
  "$(terraform output -raw curated_bucket)" \
  "$(terraform output -raw raw_bucket)"
```

The App Runner **instance role** already has read access to the curated and
artifacts buckets, so a future S3-backed storage path needs no IAM changes.

---

## 7. Continuous deployment (GitHub Actions)

[`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) builds and pushes
the image on every push to `main`. App Runner has `auto_deployments_enabled`, so a
new image on the `latest` tag triggers a rolling redeploy automatically.

Set up OIDC-based auth (no long-lived keys):

1. Create an IAM role trusting GitHub's OIDC provider
   (`token.actions.githubusercontent.com`) with permission to push to ECR
   (`AmazonEC2ContainerRegistryPowerUser` or a scoped policy).
2. Add the role ARN as the repo secret **`AWS_DEPLOY_ROLE_ARN`**.

The workflow regenerates demo artifacts from synthetic data before building. For
production, swap that step for an `aws s3 cp` of your real checkpoint + `clean.csv`.

---

## 8. Observability

- **Logs** — App Runner streams application + service logs to CloudWatch Logs
  (`/aws/apprunner/<service>/…`). View them in the App Runner console → *Logs*.
- **Metrics** — CPU, memory, request count, latency, and 2xx/4xx/5xx counts in the
  `AWS/AppRunner` namespace.
- **Alarms** — Terraform creates a CPU > 80% alarm and a 5xx-burst alarm. Set
  `alarm_email` in `terraform.tfvars` to receive SNS email notifications (confirm
  the subscription email AWS sends).

---

## 9. Updating the deployment

```bash
# Rebuild with new code/artifacts and push — App Runner auto-redeploys.
./scripts/deploy/build_and_push.sh

# Change infra (sizing, scaling, alarms) — edit terraform.tfvars, then:
cd infra/terraform && terraform apply
```

---

## 10. Teardown

```bash
cd infra/terraform
terraform destroy
```

`force_delete = true` on the ECR repo and S3 buckets lets Terraform remove them
even with images/objects present. **This deletes the buckets and their contents** —
back up anything you need first.

---

## Cost notes

- **App Runner** bills for provisioned + active instance time; `min_size = 1`
  keeps one instance warm (App Runner requires ≥1). To fully stop billing, pause
  the service in the console or run `terraform destroy`.
- **ECR / S3** bill for stored GB; the ECR lifecycle policy caps image history at
  10, and S3 versioning is on (old versions accrue storage).
- The image is large (~3–4 GB; torch + transformers + tsfresh), which lengthens
  build/deploy time but does not change runtime cost.
