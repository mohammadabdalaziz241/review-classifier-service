#!/bin/bash
# Pull the released image and (re)start the stack. Run by systemd at every boot and
# by the Deploy workflow through Session Manager. Safe to run repeatedly: unchanged
# containers are left running.
set -euo pipefail
cd /opt/review-classifier
# shellcheck source=/dev/null
source ./instance.env # AWS_REGION, PARAM_PREFIX, LOG_GROUP

param() {
  aws ssm get-parameter --region "$AWS_REGION" --name "$PARAM_PREFIX/$1" "${@:2}" \
    --query Parameter.Value --output text
}

image=$(param image)
if [ "$image" = "none" ]; then
  echo "No release yet: run the Deploy workflow in GitHub Actions."
  exit 0
fi
postgres_password=$(param postgres-password --with-decryption)

# Compose reads .env; it holds the database password, so only root may read it.
umask 077
cat > .env <<ENV
IMAGE=$image
POSTGRES_PASSWORD=$postgres_password
AWS_REGION=$AWS_REGION
LOG_GROUP=$LOG_GROUP
ENV

registry=${image%%/*}
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$registry" > /dev/null
docker compose pull --quiet
docker compose up --detach --remove-orphans
docker image prune --force > /dev/null
echo "Running $image"
