"""Generate the Grafana dashboard (dashboards/review-classifier.json) from code.

    python monitoring/grafana/build_dashboard.py

Edit this file, not the JSON: the panels and their queries are easier to review
here, and a test checks that the JSON is up to date and that every query uses
metrics the service actually exports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "dashboards" / "review-classifier.json"
DS = {"type": "prometheus", "uid": "prometheus"}
PREDICT = 'route=~"/v1/predict.*"'
API = 'route=~"/v1/.*"'


def target(expr: str, legend: str = "", instant: bool = False) -> dict[str, Any]:
    t: dict[str, Any] = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ""}
    if instant:
        t |= {"instant": True, "range": False}
    return t


class Layout:
    """Places panels left to right on Grafana's 24-column grid."""

    def __init__(self) -> None:
        self.panels: list[dict[str, Any]] = []
        self.x = 0
        self.y = 0
        self.row_height = 0

    def row(self, title: str) -> None:
        self._newline()
        self.panels.append(
            {
                "id": len(self.panels) + 1,
                "type": "row",
                "title": title,
                "collapsed": False,
                "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1},
                "panels": [],
            }
        )
        self.y += 1

    def _newline(self) -> None:
        if self.x:
            self.y += self.row_height
        self.x, self.row_height = 0, 0

    def add(self, panel: dict[str, Any], w: int, h: int) -> None:
        if self.x + w > 24:
            self._newline()
        panel["gridPos"] = {"x": self.x, "y": self.y, "w": w, "h": h}
        panel["id"] = len(self.panels) + 1
        for i, t in enumerate(panel.get("targets", [])):
            t["refId"] = chr(ord("A") + i)
        self.panels.append(panel)
        self.x += w
        self.row_height = max(self.row_height, h)


def stat(
    title: str,
    targets: list[dict],
    unit: str = "short",
    description: str = "",
    mappings: list | None = None,
    thresholds: list | None = None,
    text_mode: str = "auto",
    decimals: int | None = None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "unit": unit,
        "mappings": mappings or [],
        "thresholds": {
            "mode": "absolute",
            "steps": thresholds or [{"color": "green", "value": None}],
        },
        "color": {"mode": "thresholds"},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": text_mode,
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto",
        },
    }


def timeseries(
    title: str,
    targets: list[dict],
    unit: str = "short",
    description: str = "",
    stacked: bool = False,
    percent: bool = False,
    min_: float | None = 0,
    max_: float | None = None,
) -> dict[str, Any]:
    custom: dict[str, Any] = {
        "drawStyle": "line",
        "lineWidth": 1,
        "fillOpacity": 25 if stacked else 10,
        "showPoints": "never",
        "spanNulls": False,
        "stacking": {"mode": "percent" if percent else "normal" if stacked else "none"},
    }
    defaults: dict[str, Any] = {"unit": unit, "custom": custom}
    if min_ is not None:
        defaults["min"] = min_
    if max_ is not None:
        defaults["max"] = max_
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": DS,
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def quantile(q: float, metric: str, selector: str = "", by: str = "") -> str:
    group = f"le{', ' + by if by else ''}"
    return (
        f"histogram_quantile({q}, sum by ({group}) "
        f"(rate({metric}_bucket{{{selector}}}[$__rate_interval])))"
    )


def build() -> dict[str, Any]:
    g = Layout()

    # ---- overview ---------------------------------------------------------------
    g.row("Overview")
    g.add(
        stat(
            "Model",
            [target("model_info", "{{model_id}} @ {{version}}", instant=True)],
            text_mode="name",
            description="The model the API has loaded, and its exact version.",
        ),
        w=8,
        h=4,
    )
    g.add(
        stat(
            "Model ready",
            [target("model_ready", instant=True)],
            mappings=[
                {
                    "type": "value",
                    "options": {
                        "0": {"text": "Not ready", "color": "red"},
                        "1": {"text": "Ready", "color": "green"},
                    },
                }
            ],
        ),
        w=3,
        h=4,
    )
    g.add(
        stat(
            "Database",
            [target("db_up", instant=True)],
            description="Whether predictions are being recorded. No value: no database.",
            mappings=[
                {
                    "type": "value",
                    "options": {
                        "0": {"text": "Unavailable", "color": "orange"},
                        "1": {"text": "Recording", "color": "green"},
                    },
                },
                {"type": "special", "options": {"match": "null", "result": {"text": "Off"}}},
            ],
        ),
        w=3,
        h=4,
    )
    g.add(
        stat(
            "Requests / s",
            [target(f"sum(rate(http_requests_total{{{API}}}[$__rate_interval])) or vector(0)")],
            unit="reqps",
            decimals=2,
        ),
        w=3,
        h=4,
    )
    g.add(
        stat(
            "Server errors",
            [
                target(
                    f'(sum(rate(http_requests_total{{{API}, status=~"5.."}}[$__rate_interval]))'
                    " or vector(0))"
                    f" / sum(rate(http_requests_total{{{API}}}[$__rate_interval]))"
                )
            ],
            unit="percentunit",
            description="Share of API requests answered with a 5xx status.",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.01},
                {"color": "red", "value": 0.05},
            ],
        ),
        w=3,
        h=4,
    )
    g.add(
        stat(
            "p95 latency",
            [target(quantile(0.95, "http_request_duration_seconds", 'route="/v1/predict"'))],
            unit="s",
            description="95th percentile time to answer POST /v1/predict, inside the server.",
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.5},
                {"color": "red", "value": 1},
            ],
        ),
        w=4,
        h=4,
    )

    # ---- traffic and latency ----------------------------------------------------
    g.row("Traffic and latency")
    g.add(
        timeseries(
            "Requests by route",
            [
                target(
                    f"sum by (route) (rate(http_requests_total{{{API}}}[$__rate_interval]))",
                    "{{route}}",
                )
            ],
            unit="reqps",
        ),
        w=8,
        h=8,
    )
    g.add(
        timeseries(
            "Responses by status",
            [
                target(
                    f"sum by (status) (rate(http_requests_total{{{API}}}[$__rate_interval]))",
                    "{{status}}",
                )
            ],
            unit="reqps",
            stacked=True,
            description="4xx are client errors (bad input); 5xx are the service's own failures.",
        ),
        w=8,
        h=8,
    )
    g.add(
        timeseries(
            "Latency of POST /v1/predict",
            [
                target(
                    quantile(q, "http_request_duration_seconds", 'route="/v1/predict"'),
                    f"p{int(q * 100)}",
                )
                for q in (0.5, 0.95, 0.99)
            ],
            unit="s",
            description="Time inside the server, from receiving the request to answering it.",
        ),
        w=8,
        h=8,
    )
    g.add(
        timeseries(
            "Model time and queue wait (p95)",
            [
                target(
                    quantile(0.95, "model_inference_duration_seconds", by="endpoint"),
                    "model: {{endpoint}}",
                ),
                target(quantile(0.95, "model_inference_queue_seconds"), "waiting for a slot"),
            ],
            unit="s",
            description=(
                "Model time is the forward pass alone. Waiting grows when requests arrive "
                "faster than the model can serve them (MAX_CONCURRENT_INFERENCES)."
            ),
        ),
        w=12,
        h=8,
    )
    g.add(
        timeseries(
            "Requests in progress",
            [target("http_requests_in_progress", "in progress")],
            description="Includes the Prometheus scrape itself.",
        ),
        w=6,
        h=8,
    )
    g.add(
        timeseries(
            "Texts per forward pass",
            [
                target(
                    "sum(rate(model_batch_texts_sum[$__rate_interval]))"
                    " / sum(rate(model_batch_texts_count[$__rate_interval]))",
                    "average",
                )
            ],
            description=(
                "With dynamic batching (BATCH_REQUESTS), requests that wait for the model "
                "share a pass, so this rises with load. Without it, it is the texts per request."
            ),
        ),
        w=6,
        h=8,
    )

    # ---- model behaviour --------------------------------------------------------
    g.row("Model behaviour")
    g.add(
        timeseries(
            "Predicted labels",
            [
                target(
                    "sum by (label) (rate(model_predictions_total[$__rate_interval]))",
                    "{{label}}",
                )
            ],
            unit="percentunit",
            stacked=True,
            percent=True,
            description=(
                "Share of each predicted label. A lasting shift can mean the input changed."
            ),
        ),
        w=8,
        h=8,
    )
    g.add(
        timeseries(
            "Confidence of predictions",
            [
                target(quantile(0.5, "model_prediction_confidence"), "median"),
                target(quantile(0.1, "model_prediction_confidence"), "lowest 10%"),
            ],
            unit="percentunit",
            min_=0.5,
            max_=1,
            description="Probability of the predicted label. Falling confidence can signal drift.",
        ),
        w=8,
        h=8,
    )
    g.add(
        timeseries(
            "Input length (characters)",
            [
                target(quantile(0.5, "model_input_chars"), "median"),
                target(quantile(0.95, "model_input_chars"), "p95"),
            ],
            description="Long texts cost more: RoBERTa reads up to 256 tokens of each.",
        ),
        w=8,
        h=8,
    )
    g.add(
        stat(
            "Accuracy from feedback",
            [
                target(
                    'sum(increase(feedback_total{model_correct="true"}[$__range]))'
                    " / sum(increase(feedback_total[$__range]))"
                )
            ],
            unit="percentunit",
            description="Share of predictions users confirmed as correct, over the time range.",
            thresholds=[
                {"color": "red", "value": None},
                {"color": "orange", "value": 0.7},
                {"color": "green", "value": 0.85},
            ],
        ),
        w=6,
        h=5,
    )
    g.add(
        stat(
            "Feedback received",
            [target("sum(increase(feedback_total[$__range])) or vector(0)")],
            decimals=0,
            description=(
                "Over the time range. Prometheus estimates increases from 15-second samples, "
                "so small counts are approximate."
            ),
        ),
        w=6,
        h=5,
    )
    g.add(
        timeseries(
            "Predictions recorded in the database",
            [
                target(
                    "sum by (outcome) (rate(db_prediction_records_total[$__rate_interval]))",
                    "{{outcome}}",
                )
            ],
            unit="reqps",
            stacked=True,
            description=(
                "not_recorded: the database was unavailable; predictions were still served."
            ),
        ),
        w=12,
        h=5,
    )

    # ---- resources --------------------------------------------------------------
    g.row("Resources")
    g.add(
        timeseries(
            "API memory",
            [target('process_resident_memory_bytes{job="api"}', "resident memory")],
            unit="bytes",
        ),
        w=6,
        h=8,
    )
    g.add(
        timeseries(
            "CPU",
            [
                target('rate(process_cpu_seconds_total{job="api"}[$__rate_interval])', "API"),
                target(
                    'sum(rate(node_cpu_seconds_total{mode!="idle"}[$__rate_interval]))',
                    "whole host",
                ),
                target('count(node_cpu_seconds_total{mode="idle"})', "cores available"),
            ],
            unit="short",
            description="In cores: 1 means one CPU core fully busy.",
        ),
        w=6,
        h=8,
    )
    g.add(
        timeseries(
            "Host memory and disk used",
            [
                target("1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes", "memory"),
                target(
                    '1 - node_filesystem_avail_bytes{mountpoint="/", fstype!~"tmpfs|overlay"}'
                    ' / node_filesystem_size_bytes{mountpoint="/", fstype!~"tmpfs|overlay"}',
                    "root disk",
                ),
            ],
            unit="percentunit",
            max_=1,
        ),
        w=6,
        h=8,
    )
    g.add(
        {
            "type": "table",
            "title": "Firing alerts",
            "description": "Alert rules from monitoring/prometheus/alerts.yml that are firing now.",
            "datasource": DS,
            "targets": [target('ALERTS{alertstate="firing"}', instant=True) | {"format": "table"}],
            "fieldConfig": {"defaults": {}, "overrides": []},
            "options": {"showHeader": True},
            "transformations": [
                {
                    "id": "organize",
                    "options": {
                        "excludeByName": {"Time": True, "Value": True, "__name__": True},
                        "renameByName": {"alertname": "Alert", "severity": "Severity"},
                    },
                }
            ],
        },
        w=6,
        h=8,
    )

    return {
        "uid": "review-classifier",
        "title": "Review classifier service",
        "description": "Traffic, latency, model behaviour and resources of the review API.",
        "tags": ["review-classifier"],
        "timezone": "browser",
        "editable": False,
        "graphTooltip": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "schemaVersion": 39,
        "version": 1,
        "templating": {"list": []},
        "annotations": {"list": []},
        "panels": g.panels,
    }


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT}")
