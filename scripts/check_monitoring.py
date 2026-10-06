#!/usr/bin/env python3
"""Check that the monitoring stack works: Prometheus scrapes, Grafana serves the dashboard.

    docker compose --profile monitoring up --detach
    python scripts/check_monitoring.py

Waits until Prometheus reports the API and node exporter targets as up and has
the service's metrics, then checks that Grafana is healthy, has provisioned the
dashboard, and can query Prometheus through its data source, as the dashboard
does. Standard library only. Exit code 0 when everything passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_for(description: str, check, timeout: float) -> object:
    deadline = time.monotonic() + timeout
    last: object = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                print(f"  ok  {description}")
                return result
            last = result
        except (OSError, ValueError, KeyError) as exc:
            last = exc
        time.sleep(3)
    raise SystemExit(f"FAIL {description} (last: {last})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--grafana", default="http://127.0.0.1:3000")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    prometheus, grafana = args.prometheus.rstrip("/"), args.grafana.rstrip("/")

    def targets_up() -> bool:
        active = get(f"{prometheus}/api/v1/targets")["data"]["activeTargets"]
        health = {t["labels"]["job"]: t["health"] for t in active}
        return health.get("api") == "up" and health.get("node") == "up"

    def query(expr: str) -> list:
        url = f"{prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
        return get(url)["data"]["result"]

    wait_for("Prometheus scrapes the API and the node exporter", targets_up, args.timeout)
    wait_for("the model is reported ready", lambda: query("model_ready == 1"), args.timeout)
    wait_for("API traffic is recorded", lambda: query("sum(http_requests_total) > 0"), args.timeout)
    rules = get(f"{prometheus}/api/v1/rules")["data"]["groups"]
    broken = [r["name"] for g in rules for r in g["rules"] if r.get("health") == "err"]
    if not rules or broken:
        raise SystemExit(f"FAIL alert rules loaded without errors (broken: {broken})")
    print(f"  ok  {sum(len(g['rules']) for g in rules)} alert rules loaded")

    wait_for("Grafana is healthy", lambda: get(f"{grafana}/api/health")["database"] == "ok", 60)
    dashboard = wait_for(
        "Grafana has provisioned the dashboard",
        lambda: get(f"{grafana}/api/dashboards/uid/review-classifier")["dashboard"],
        60,
    )
    panels = [p for p in dashboard["panels"] if p.get("type") != "row"]
    result = post(
        f"{grafana}/api/ds/query",
        {
            "from": "now-5m",
            "to": "now",
            "queries": [{"refId": "A", "datasource": {"uid": "prometheus"}, "expr": "model_ready"}],
        },
    )
    frames = result["results"]["A"].get("frames", [])
    if not frames:
        raise SystemExit(f"FAIL Grafana queries Prometheus: {result}")
    print(f"  ok  Grafana queries Prometheus; the dashboard has {len(panels)} panels")
    print("Monitoring check passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        sys.exit(f"FAIL {exc.url}: HTTP {exc.code}")
