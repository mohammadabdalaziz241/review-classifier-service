"""The dashboard and alert rules only use metrics the service exports."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from review_classifier.api import create_app
from review_classifier.config import Settings

ROOT = Path(__file__).resolve().parents[1]
MONITORING = ROOT / "monitoring"
DASHBOARD = MONITORING / "grafana" / "dashboards" / "review-classifier.json"
ALERTS = MONITORING / "prometheus" / "alerts.yml"
# Metric families of this service; others (node_*, up, ALERTS) come from Prometheus
# and the node exporter.
OWN = re.compile(r"\b((?:http|model|db|feedback|process)_[a-z_]+|feedback_total)\b")


def _metric_names(text: str) -> set[str]:
    """Service metric names in PromQL, ignoring label matchers ({model_correct="true"})."""
    return set(OWN.findall(re.sub(r"\{[^}]*\}", "", text)))


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_dashboard", MONITORING / "grafana" / "build_dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> set[str]:
    """Sample names in a scrape of a service with a database, after some traffic."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("alembic")
    from review_classifier.db import create_db_engine, upgrade

    url = f"sqlite:///{tmp_path_factory.mktemp('db') / 'm.db'}"
    engine = create_db_engine(url)
    upgrade(engine)
    engine.dispose()
    with TestClient(create_app(Settings(model_backend="dummy", database_url=url))) as client:
        p = client.post("/v1/predict", json={"text": "great"}).json()["prediction"]
        client.post("/v1/feedback", json={"prediction_id": p["id"], "label": "positive"})
        text = client.get("/metrics").text
    names = {line.split("{")[0].split(" ")[0] for line in text.splitlines() if line[:1].isalpha()}
    # Process metrics are Linux-only; the deployment runs on Linux.
    return names | {"process_resident_memory_bytes", "process_cpu_seconds_total"}


def _queries() -> list[str]:
    dashboard = json.loads(DASHBOARD.read_text())
    return [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]


def test_dashboard_json_is_generated_from_the_builder():
    assert DASHBOARD.read_text() == _builder().render(), (
        "Run: python monitoring/grafana/build_dashboard.py"
    )


def test_dashboard_panels_fit_the_grid():
    for panel in json.loads(DASHBOARD.read_text())["panels"]:
        pos = panel["gridPos"]
        assert pos["x"] + pos["w"] <= 24, panel["title"]
        for target in panel.get("targets", []):
            assert target["datasource"]["uid"] == "prometheus"


def test_dashboard_queries_use_exported_metrics(exported):
    used = set().union(*map(_metric_names, _queries()))
    assert used, "no service metrics found in the dashboard"
    assert used <= exported, sorted(used - exported)


def test_alert_rules_use_exported_metrics(exported):
    used = _metric_names(ALERTS.read_text())
    assert {"model_ready", "db_up"} <= used
    assert used <= exported, sorted(used - exported)
