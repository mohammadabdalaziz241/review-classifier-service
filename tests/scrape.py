"""Parse Prometheus text output in tests."""

from __future__ import annotations

from prometheus_client.parser import text_string_to_metric_families


def parse(text: str) -> dict[tuple[str, frozenset], float]:
    """{(sample name, labels): value} for every sample in a scrape."""
    return {
        (s.name, frozenset(s.labels.items())): s.value
        for family in text_string_to_metric_families(text)
        for s in family.samples
    }


def value(samples: dict, name: str, **labels: str) -> float | None:
    return samples.get((name, frozenset(labels.items())))
