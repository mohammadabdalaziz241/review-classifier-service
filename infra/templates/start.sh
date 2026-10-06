#!/bin/bash
# Start the released stack. Run by systemd at every boot and by the Deploy workflow
# through Session Manager (which first installs the latest version of this script).
# Safe to run repeatedly: unchanged containers are left running.
#
# The release image carries its own deployment files at /opt/release (Compose file and
# monitoring configuration); they are copied out here, so they always match the code.
set -euo pipefail
cd /opt/review-classifier
# One run at a time: the boot run and a deploy can overlap after a start.
exec 9> /run/review-classifier.lock
flock 9
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

registry=${image%%/*}
aws ecr get-login-password --region "$AWS_REGION" |
  docker login --username AWS --password-stdin "$registry" > /dev/null
docker pull --quiet "$image" > /dev/null

# The release's deployment files. Replaced only when they changed, so containers
# that mount them are not disturbed by a restart with the same release.
rm -rf release.new
mkdir release.new
container=$(docker create "$image")
if docker cp "$container:/opt/release/." release.new/ > /dev/null 2>&1; then
  if diff -r -q release.new release > /dev/null 2>&1; then
    rm -rf release.new
  else
    rm -rf release
    mv release.new release
  fi
  cp release/compose.yaml compose.yaml
else
  # An image from before releases carried these files: keep the current ones.
  rm -rf release.new
  docker rm "$container" > /dev/null
  if [ ! -f compose.yaml ]; then
    echo "error: $image has no deployment files and there is no compose.yaml here." >&2
    echo "Run the Deploy workflow to release a current image." >&2
    exit 1
  fi
  echo "This image has no deployment files; using the existing compose.yaml."
  container=""
fi
[ -z "$container" ] || docker rm "$container" > /dev/null

# Recreate the monitoring containers whenever the release's files change, so they
# never keep mounts of a replaced directory.
config_version=none
if [ -d release ]; then
  config_version=$(cd release && find . -type f -print0 | sort -z |
    xargs -0 sha256sum | sha256sum | cut -c1-16)
fi

# Compose reads .env; it holds the database password, so only root may read it.
umask 077
cat > .env << ENV
IMAGE=$image
POSTGRES_PASSWORD=$postgres_password
AWS_REGION=$AWS_REGION
LOG_GROUP=$LOG_GROUP
MONITORING_CONFIG_VERSION=$config_version
ENV
umask 022

docker compose pull --quiet --ignore-pull-failures
# The service first, so a registry problem with a monitoring image cannot block a release.
docker compose up --detach --remove-orphans db migrate api
if ! docker compose up --detach --remove-orphans; then
  echo "warning: the monitoring containers did not start; the API is running." >&2
fi
docker image prune --force > /dev/null
echo "Running $image"
docker compose ps --format '{{.Service}}: {{.Status}}'
