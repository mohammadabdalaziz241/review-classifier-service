data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

# Latest Amazon Linux 2023 at the time of the first apply. Later AMI releases are
# ignored (see the instance's lifecycle block) so they never replace the server.
data "aws_ssm_parameter" "al2023" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  account_id   = data.aws_caller_identity.current.account_id
  param_prefix = "/${var.project}"
  log_group    = "/${var.project}"
  subnet_id    = sort(data.aws_subnets.default.ids)[0]
}
