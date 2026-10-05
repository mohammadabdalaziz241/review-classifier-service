variable "region" {
  description = "AWS region for everything."
  type        = string
  default     = "eu-north-1"
}

variable "project" {
  description = "Name used for resources, tags, the SSM parameter prefix and the log group."
  type        = string
  default     = "review-classifier"
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, as owner/name."
  type        = string
  default     = "mohammadabdalaziz241/review-classifier-service"
}

variable "github_owner_id" {
  description = <<-EOT
    Numeric ID of the repository owner on GitHub. Repositories created after
    15 July 2026 sign their OIDC tokens with IDs as well as names; set this and
    github_repository_id for them. Leave both unset for older repositories.
  EOT
  type        = number
  default     = null
}

variable "github_repository_id" {
  description = "Numeric ID of the GitHub repository (see github_owner_id)."
  type        = number
  default     = null

  validation {
    condition     = (var.github_repository_id == null) == (var.github_owner_id == null)
    error_message = "Set both github_owner_id and github_repository_id, or neither."
  }
}

variable "github_branch" {
  description = "Only workflows running on this branch may deploy."
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = "Create the GitHub OIDC identity provider. Set false if the account already has one."
  type        = bool
  default     = true
}

variable "instance_type" {
  description = "c7i-flex.large (2 vCPU, 4 GiB) is Free plan eligible and fits the model with headroom."
  type        = string
  default     = "c7i-flex.large"
}

variable "root_volume_gb" {
  description = "Disk for the OS, Docker images and PostgreSQL data."
  type        = number
  default     = 20
}

variable "allowed_cidrs" {
  description = "Who may reach the API on port 80. 0.0.0.0/0 lets anyone (e.g. a recruiter) try it while it runs."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "idle_stop_minutes" {
  description = <<-EOT
    Stop the instance after this many minutes with CPU below idle_cpu_percent. CPU is
    averaged over 5 minutes, so light demo traffic does not keep it running: this is
    roughly the longest session before it stops itself.
  EOT
  type        = number
  default     = 120

  validation {
    condition     = var.idle_stop_minutes >= 15 && var.idle_stop_minutes <= 1440 && var.idle_stop_minutes % 5 == 0
    error_message = "idle_stop_minutes must be a multiple of 5 between 15 and 1440."
  }
}

variable "idle_cpu_percent" {
  description = "CPU (5-minute average) below which the instance counts as idle."
  type        = number
  default     = 5
}

variable "monthly_budget_usd" {
  description = "Monthly budget for the budget alert. Credits are excluded, so it tracks real usage."
  type        = number
  default     = 10
}

variable "alert_email" {
  description = "Where budget alerts are emailed."
  type        = string
}

variable "log_retention_days" {
  description = "How long container logs are kept in CloudWatch."
  type        = number
  default     = 14
}

variable "compose_version" {
  description = "Docker Compose release installed on the instance."
  type        = string
  default     = "v2.29.7"
}
