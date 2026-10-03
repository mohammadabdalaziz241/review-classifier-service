# ---- configuration the instance reads at start-up ------------------------------------

# The release to run, as an immutable image digest. The deploy workflow updates it,
# so Terraform only creates it and never resets it.
resource "aws_ssm_parameter" "image" {
  # checkov:skip=CKV2_AWS_34: An image reference is not a secret.
  name        = "${local.param_prefix}/image"
  description = "Image the instance runs: <registry>/<repo>@sha256:<digest>, or 'none'."
  type        = "String"
  value       = "none"

  lifecycle {
    ignore_changes = [value]
  }
}

resource "random_password" "postgres" {
  length  = 32
  special = false
}

resource "aws_ssm_parameter" "postgres_password" {
  # checkov:skip=CKV_AWS_337: Encrypted with the AWS-managed SSM key; a customer-managed key costs $1/month.
  name        = "${local.param_prefix}/postgres-password"
  description = "Password of the PostgreSQL user on the instance."
  type        = "SecureString"
  value       = random_password.postgres.result
}

resource "aws_cloudwatch_log_group" "app" {
  # checkov:skip=CKV_AWS_158: Encrypted by CloudWatch by default; logs contain no review text.
  # checkov:skip=CKV_AWS_338: 14 days is enough to debug a demo service and keeps storage cost near zero.
  name              = local.log_group
  retention_in_days = var.log_retention_days
}

# ---- what the instance may do ------------------------------------------------------

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.project}-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

# Session Manager: shell access and remote commands without SSH keys or port 22.
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "instance" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "PullTheAppImage"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.app.arn]
  }
  statement {
    sid     = "ReadOwnConfiguration"
    actions = ["ssm:GetParameter", "ssm:GetParameters"]
    resources = [
      aws_ssm_parameter.image.arn,
      aws_ssm_parameter.postgres_password.arn,
    ]
  }
  statement {
    sid       = "WriteContainerLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${aws_cloudwatch_log_group.app.arn}:*"]
  }
}

resource "aws_iam_role_policy" "instance" {
  name   = "${var.project}-instance"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance.json
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.project}-instance"
  role = aws_iam_role.instance.name
}

# ---- network: only the API port is open --------------------------------------------

resource "aws_security_group" "instance" {
  name        = "${var.project}-instance"
  description = "API on port 80; no SSH (use Session Manager)"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_ingress_rule" "http" {
  # checkov:skip=CKV_AWS_260: Public demo API by design; narrow with allowed_cidrs.
  for_each          = toset(var.allowed_cidrs)
  security_group_id = aws_security_group.instance.id
  description       = "API"
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  cidr_ipv4         = each.value
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.instance.id
  description       = "Image pulls, AWS APIs, OS updates"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

# ---- the server ----------------------------------------------------------------------

resource "aws_instance" "app" {
  # checkov:skip=CKV_AWS_135: Current-generation instance types are EBS-optimized by default.
  # checkov:skip=CKV_AWS_88: The public IP replaces a load balancer (about $18/month) for a single on-demand instance.
  # checkov:skip=CKV_AWS_126: Basic 5-minute metrics drive the idle alarm; detailed monitoring costs extra.
  ami                         = data.aws_ssm_parameter.al2023.value
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.instance.id]
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  associate_public_ip_address = true

  # IMDSv2 only, and one network hop: the host can read its role credentials,
  # containers cannot.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/templates/user_data.sh.tftpl", {
    region          = var.region
    param_prefix    = local.param_prefix
    log_group       = local.log_group
    compose_version = var.compose_version
    compose_b64     = base64encode(file("${path.module}/templates/compose.yaml"))
    start_b64       = base64encode(file("${path.module}/templates/start.sh"))
    unit_b64        = base64encode(file("${path.module}/templates/review-classifier.service"))
  })
  user_data_replace_on_change = false

  tags = {
    Name = var.project
  }

  lifecycle {
    ignore_changes = [ami]
  }

  depends_on = [aws_iam_role_policy.instance]
}

# ---- cost guard: stop when idle --------------------------------------------------------

# If the CPU stays below idle_cpu_percent for idle_stop_minutes, stop the instance.
# A forgotten instance then costs about an hour, not a month.
resource "aws_cloudwatch_metric_alarm" "idle_stop" {
  alarm_name          = "${var.project}-idle-stop"
  alarm_description   = "Stops ${var.project} after ${var.idle_stop_minutes} idle minutes."
  namespace           = "AWS/EC2"
  metric_name         = "CPUUtilization"
  dimensions          = { InstanceId = aws_instance.app.id }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = var.idle_stop_minutes / 5
  threshold           = var.idle_cpu_percent
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = ["arn:${data.aws_partition.current.partition}:automate:${var.region}:ec2:stop"]
}
