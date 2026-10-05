# GitHub Actions signs in with a short-lived OIDC token: no AWS keys stored in GitHub.

resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_github_oidc_provider ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # AWS validates GitHub's certificate itself; the thumbprints are required by
  # older provider versions only.
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  github_oidc_arn = (var.create_github_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
  : data.aws_iam_openid_connect_provider.github[0].arn)

  # The token's subject names the repository and branch. Repositories created after
  # 15 July 2026 use the immutable form, with the owner and repository IDs, so a
  # repository later re-created under the same name cannot deploy:
  #   repo:OWNER@OWNER_ID/REPO@REPO_ID:ref:refs/heads/BRANCH
  # Older repositories use repo:OWNER/REPO:ref:refs/heads/BRANCH.
  github_owner = split("/", var.github_repository)[0]
  github_repo  = split("/", var.github_repository)[1]
  github_subject = (var.github_repository_id == null
    ? "repo:${var.github_repository}:ref:refs/heads/${var.github_branch}"
    : "repo:${local.github_owner}@${var.github_owner_id}/${local.github_repo}@${var.github_repository_id}:ref:refs/heads/${var.github_branch}"
  )
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only workflows of this repository, on this branch.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [local.github_subject]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name                 = "${var.project}-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_assume.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "PushTheAppImage"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = [aws_ecr_repository.app.arn]
  }
  statement {
    sid       = "RecordTheRelease"
    actions   = ["ssm:GetParameter", "ssm:PutParameter"]
    resources = [aws_ssm_parameter.image.arn]
  }
  statement {
    sid       = "FindTheInstance"
    actions   = ["ec2:DescribeInstances"]
    resources = ["*"]
  }
  statement {
    sid     = "RollOutOnTheInstance"
    actions = ["ssm:SendCommand"]
    resources = [
      aws_instance.app.arn,
      "arn:${data.aws_partition.current.partition}:ssm:${var.region}::document/AWS-RunShellScript",
    ]
  }
  statement {
    sid       = "FollowTheRollOut"
    actions   = ["ssm:GetCommandInvocation"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "${var.project}-github-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}
