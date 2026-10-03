#!/bin/bash
# Start, stop and inspect the AWS deployment.
#
#   scripts/aws.sh start     start the instance, wait until the API is ready, print its URL
#   scripts/aws.sh stop      stop the instance (disk and data are kept)
#   scripts/aws.sh status    state, URL, running time and its approximate cost, release
#   scripts/aws.sh logs      recent container logs (add -f to follow)
#   scripts/aws.sh shell     a root shell on the instance via Session Manager (no SSH)
#
# Uses your AWS CLI credentials (AWS_PROFILE) and AWS_REGION (default eu-north-1).
# Works with the bash 3.2 that ships with macOS.
set -euo pipefail

PROJECT=review-classifier
REGION=${AWS_REGION:-eu-north-1}
# c7i-flex.large in eu-north-1 ($0.0908/h) + its public IPv4 address ($0.005/h).
HOURLY_USD=0.0958
IDLE_MINUTES=120

die() {
  echo "error: $*" >&2
  exit 1
}

command -v aws > /dev/null || die "the AWS CLI is not installed (brew install awscli)"

aws_() { aws --region "$REGION" "$@"; }

instance_id() {
  local id
  id=$(aws_ ec2 describe-instances \
    --filters "Name=tag:Project,Values=$PROJECT" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -n "$id" ] && [ "$id" != "None" ] || die "no $PROJECT instance found in $REGION (run terraform apply in infra/)"
  echo "$id"
}

field() { # field <instance-id> <query>
  aws_ ec2 describe-instances --instance-ids "$1" --query "Reservations[0].Instances[0].$2" --output text
}

seconds_since() { # ISO-8601 timestamp -> seconds elapsed (GNU or BSD date)
  local start now
  now=$(date -u +%s)
  start=$(date -u -d "$1" +%s 2> /dev/null ||
    date -u -j -f "%Y-%m-%dT%H:%M:%S" "${1%%[.+]*}" +%s 2> /dev/null ||
    echo "$now")
  echo $((now - start))
}

release() {
  aws_ ssm get-parameter --name "/$PROJECT/image" --query Parameter.Value --output text 2> /dev/null ||
    echo "unknown"
}

wait_ready() { # wait_ready <url>
  printf "Waiting for the API"
  for _ in $(seq 1 60); do
    if curl -fsS -m 5 "$1/ready" > /dev/null 2>&1; then
      echo " ready."
      return 0
    fi
    printf "."
    sleep 10
  done
  echo
  die "the API did not become ready within 10 minutes; check: scripts/aws.sh logs"
}

cmd_start() {
  local id state ip url
  id=$(instance_id)
  state=$(field "$id" State.Name)
  case "$state" in
    running) echo "Already running." ;;
    stopping)
      echo "The instance is still stopping; waiting before starting it again..."
      aws_ ec2 wait instance-stopped --instance-ids "$id"
      ;;
  esac
  if [ "$state" != "running" ]; then
    echo "Starting $id..."
    aws_ ec2 start-instances --instance-ids "$id" > /dev/null
    aws_ ec2 wait instance-running --instance-ids "$id"
  fi
  if [ "$(release)" = "none" ]; then
    echo "The instance is running, but nothing is deployed yet."
    echo "Run the Deploy workflow in GitHub Actions; it will roll out to this instance."
    return 0
  fi
  ip=$(field "$id" PublicIpAddress)
  url="http://$ip"
  wait_ready "$url"
  echo
  echo "  API:   $url"
  echo "  Docs:  $url/docs"
  echo "  Model: $(curl -fsS -m 5 "$url/v1/model" 2> /dev/null | sed -n 's/.*"id":"\([^"]*\)".*"version":"\([^"]*\)".*/\1 @ \2/p')"
  echo
  echo "It costs about \$$HOURLY_USD per hour while running and stops itself after"
  echo "$IDLE_MINUTES idle minutes. Stop it now with: scripts/aws.sh stop"
}

cmd_stop() {
  local id state
  id=$(instance_id)
  state=$(field "$id" State.Name)
  if [ "$state" = "stopped" ]; then
    echo "Already stopped."
    return 0
  fi
  report_session "$id"
  echo "Stopping $id..."
  aws_ ec2 stop-instances --instance-ids "$id" > /dev/null
  aws_ ec2 wait instance-stopped --instance-ids "$id"
  echo "Stopped. Disk and database are kept; only storage (about \$2/month) is charged."
}

report_session() { # running time and its approximate cost
  local launched seconds
  launched=$(field "$1" LaunchTime)
  seconds=$(seconds_since "$launched")
  awk -v s="$seconds" -v rate="$HOURLY_USD" \
    'BEGIN { printf "Running for %dh%02dm, about $%.2f this session.\n", s/3600, (s%3600)/60, s/3600*rate }'
}

cmd_status() {
  local id state ip
  id=$(instance_id)
  state=$(field "$id" State.Name)
  echo "Instance: $id ($(field "$id" InstanceType)) in $REGION"
  echo "State:    $state"
  if [ "$state" = "running" ]; then
    ip=$(field "$id" PublicIpAddress)
    echo "API:      http://$ip  (ready: $(curl -fsS -m 5 "http://$ip/ready" 2> /dev/null || echo no))"
    printf "Session:  "
    report_session "$id"
  fi
  echo "Release:  $(release)"
  echo "Credits:  https://console.aws.amazon.com/billing/home#/freetier"
}

cmd_logs() {
  local follow=""
  [ "${1:-}" = "-f" ] && follow="--follow"
  # shellcheck disable=SC2086
  aws_ logs tail "/$PROJECT" --since 30m --format short $follow
}

cmd_shell() {
  command -v session-manager-plugin > /dev/null ||
    die "install the Session Manager plugin: brew install --cask session-manager-plugin"
  aws_ ssm start-session --target "$(instance_id)"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) cmd_logs "${2:-}" ;;
  shell) cmd_shell ;;
  *)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
