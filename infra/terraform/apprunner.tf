# Auto-scaling policy: how aggressively App Runner adds/removes instances.
resource "aws_apprunner_auto_scaling_configuration_version" "app" {
  # App Runner caps this name at 32 chars, so use the (shorter) project name
  # rather than the "<project>-<environment>" prefix used elsewhere.
  auto_scaling_configuration_name = var.project_name
  max_concurrency                 = var.max_concurrency
  min_size                        = var.min_size
  max_size                        = var.max_size

  lifecycle {
    create_before_destroy = true
  }
}

# The serving service itself: pulls the image from ECR and exposes an HTTPS URL.
resource "aws_apprunner_service" "app" {
  service_name = local.name_prefix

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_access.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"

      image_configuration {
        port = tostring(var.container_port)

        # Surfaced to the container; the S3 buckets are here for the S3-backed
        # storage path even though the baked-in artifacts work without them.
        runtime_environment_variables = {
          AWS_REGION       = var.aws_region
          CURATED_BUCKET   = aws_s3_bucket.curated.bucket
          ARTIFACTS_BUCKET = aws_s3_bucket.artifacts.bucket
          PYTHONUNBUFFERED = "1"
        }
      }
    }

    # Redeploy automatically whenever a new image lands on the watched tag —
    # this is what makes the GitHub Actions "push to ECR" step a full deploy.
    auto_deployments_enabled = true
  }

  instance_configuration {
    cpu               = var.app_runner_cpu
    memory            = var.app_runner_memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  # Hit the FastAPI health/index route ("/" returns {"status": "ok"}).
  health_check_configuration {
    protocol            = "HTTP"
    path                = "/"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.app.arn

  depends_on = [aws_iam_role_policy_attachment.apprunner_ecr_access]
}
