resource "aws_ecr_repository" "app" {
  # checkov:skip=CKV_AWS_136: AES-256 encryption at rest is on; a customer-managed KMS key adds cost for no benefit here.
  name                 = var.project
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true # terraform destroy also removes the images

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep the last 3 images: enough to roll back, little storage to pay for.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 3 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}
