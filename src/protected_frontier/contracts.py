from __future__ import annotations

from typing import Any, Mapping


LEVELS = (0.0, 0.5, 0.75, 1.0)
DELAYS_MS = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
GOAL_IDS = ("safety_first", "quality_first", "throughput_first")
LOSS_COMPONENTS = (
    "safety_loss",
    "quality_loss",
    "defect_escape",
    "downtime",
    "traceability_loss",
    "throughput_loss",
)


def goal_specs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["goal_id"]): dict(item) for item in config["goals"]}
