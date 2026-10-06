#!/bin/bash
# Start, stop and inspect the AWS deployment.
#
#   scripts/aws.sh start     start the instance, wait until the API is ready, print its URL
#   scripts/aws.sh stop      stop the instance (disk and data are kept)
#   scripts/aws.sh status    state, URL, running time and its approximate cost, release
#   scripts/aws.sh logs      recent container logs (add -f to follow)
#   scripts/aws.sh shell     a root shell on the instance via Session Manager (no SSH)
#   scripts/aws.sh dashboard Grafana at http://localhost:3000 through a Session Manager tunnel
#   scripts/aws.sh benchmark load-test the API on the instance; saves results/benchmarks/
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

need_plugin() {
  command -v session-manager-plugin > /dev/null ||
    die "install the Session Manager plugin: brew install --cask session-manager-plugin"
}

running_instance() {
  local id
  id=$(instance_id)
  [ "$(field "$id" State.Name)" = "running" ] || die "the instance is not running: scripts/aws.sh start"
  echo "$id"
}

cmd_shell() {
  need_plugin
  aws_ ssm start-session --target "$(instance_id)"
}

cmd_dashboard() {
  local id port
  need_plugin
  id=$(running_instance)
  port=${DASHBOARD_PORT:-3000}
  echo "Grafana: http://localhost:$port  (read-only; press Ctrl+C to close the tunnel)"
  aws_ ssm start-session --target "$id" --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"3000\"],\"localPortNumber\":[\"$port\"]}"
}

# Runs the benchmark in a container on the instance, next to the API, so the network
# between them is not part of the measurement. Extra arguments are passed on, e.g.
#   scripts/aws.sh benchmark --duration 60 --scenario short-c4
cmd_benchmark() {
  local id type label command_id status output out stamp
  id=$(running_instance)
  type=$(field "$id" InstanceType)
  label="AWS $type ($REGION), client on the same instance"
  stamp=$(date -u +%Y%m%d-%H%M)
  out="results/benchmarks/aws-$type-$stamp.json"
  # The image is the deployed release; the API's metrics port is reachable inside
  # the stack's network only.
  command="set -e; cd /opt/review-classifier; . ./.env; rm -rf /tmp/bench; mkdir -m 777 /tmp/bench"
  command="$command; docker run --rm --network review-classifier_default -v /tmp/bench:/out"
  command="$command \$IMAGE python -m review_classifier.benchmark --url http://api:8000"
  command="$command --metrics-url http://api:9000/metrics --label '$label' --out /out/b.json $*"
  command="$command; echo ===JSON===; cat /tmp/bench/b.json"
  case "$command" in *\"* | *\\*) die "benchmark arguments may not contain quotes or backslashes" ;; esac
  echo "Running the benchmark on $id ($type); the standard set takes about 4 minutes..."
  command_id=$(aws_ ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
    --comment "benchmark" --timeout-seconds 600 \
    --parameters "{\"commands\":[\"$command\"],\"executionTimeout\":[\"3600\"]}" \
    --query Command.CommandId --output text)
  status=Pending
  for _ in $(seq 1 180); do
    sleep 10
    status=$(aws_ ssm get-command-invocation --command-id "$command_id" --instance-id "$id" \
      --query Status --output text 2> /dev/null || echo Pending)
    case "$status" in Pending | InProgress | Delayed) printf "." ;; *) break ;; esac
  done
  echo
  output=$(aws_ ssm get-command-invocation --command-id "$command_id" --instance-id "$id" \
    --query StandardOutputContent --output text)
  if [ "$status" != "Success" ]; then
    echo "$output"
    aws_ ssm get-command-invocation --command-id "$command_id" --instance-id "$id" \
      --query StandardErrorContent --output text >&2
    die "the benchmark ended with status $status"
  fi
  mkdir -p results/benchmarks
  printf '%s\n' "$output" | sed '/^===JSON===$/,$d'
  printf '%s\n' "$output" | sed '1,/^===JSON===$/d' > "$out"
  echo
  if ! python3 -m json.tool "$out" > /dev/null 2>&1; then
    mv "$out" "$out.txt"
    die "the report is incomplete (output longer than SSM returns?); raw output in $out.txt"
  fi
  echo "Saved $out"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) cmd_logs "${2:-}" ;;
  shell) cmd_shell ;;
  dashboard) cmd_dashboard ;;
  benchmark)
    shift
    cmd_benchmark "$@"
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
