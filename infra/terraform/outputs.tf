output "service_url" {
  description = "Public HTTPS endpoint of the deployed App Runner service."
  value       = "https://${aws_apprunner_service.app.service_url}"
}

output "ecr_repository_url" {
  description = "ECR repository URL to tag and push the Docker image to."
  value       = aws_ecr_repository.app.repository_url
}

output "raw_bucket" {
  description = "S3 raw / landing-zone bucket name."
  value       = aws_s3_bucket.raw.bucket
}

output "curated_bucket" {
  description = "S3 curated / processed-zone bucket name."
  value       = aws_s3_bucket.curated.bucket
}

output "artifacts_bucket" {
  description = "S3 model-artifacts bucket name."
  value       = aws_s3_bucket.artifacts.bucket
}

output "apprunner_instance_role_arn" {
  description = "Instance role ARN the running container assumes (S3 + CloudWatch)."
  value       = aws_iam_role.apprunner_instance.arn
}
