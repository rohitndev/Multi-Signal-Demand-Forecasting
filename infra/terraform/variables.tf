variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Identifier-safe project name used to prefix/name resources."
  type        = string
  default     = "ds02-demand-forecasting"
}

variable "environment" {
  description = "Deployment environment (prod, staging, dev)."
  type        = string
  default     = "prod"
}

variable "container_port" {
  description = "Port the FastAPI/uvicorn process listens on inside the container."
  type        = number
  default     = 8000
}

variable "image_tag" {
  description = "ECR image tag App Runner pulls and serves."
  type        = string
  default     = "latest"
}

variable "app_runner_cpu" {
  description = "App Runner instance vCPU units (1024 = 1 vCPU). ML deps need >= 1 vCPU."
  type        = string
  default     = "1024"
}

variable "app_runner_memory" {
  description = "App Runner instance memory in MB. Torch import is memory-heavy; 4096 is comfortable."
  type        = string
  default     = "4096"
}

variable "min_size" {
  description = "Minimum number of App Runner instances (provisioned warm capacity)."
  type        = number
  default     = 1
}

variable "max_size" {
  description = "Maximum number of App Runner instances to scale out to."
  type        = number
  default     = 3
}

variable "max_concurrency" {
  description = "Max concurrent requests per instance before App Runner scales out."
  type        = number
  default     = 50
}

variable "alarm_email" {
  description = "Optional email for CloudWatch alarm notifications. Empty = no SNS topic/subscription."
  type        = string
  default     = ""
}
