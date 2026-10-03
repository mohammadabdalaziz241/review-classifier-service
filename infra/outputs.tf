output "instance_id" {
  value = aws_instance.app.id
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "github_deploy_role_arn" {
  description = "Set as the AWS_ROLE_ARN variable of the GitHub repository."
  value       = aws_iam_role.github_deploy.arn
}

output "region" {
  value = var.region
}

output "log_group" {
  value = aws_cloudwatch_log_group.app.name
}

output "next_steps" {
  value = <<-EOT
    1. GitHub repository settings > Secrets and variables > Actions:
         variable AWS_ROLE_ARN = ${aws_iam_role.github_deploy.arn}
         variable AWS_REGION   = ${var.region}
         secret   HF_TOKEN     = your Hugging Face read token
    2. Actions > Deploy > Run workflow.
    3. scripts/aws.sh status   (and scripts/aws.sh stop when you are done)
  EOT
}
