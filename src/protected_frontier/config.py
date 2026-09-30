from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = (path or CONFIG_PATH).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration root is not a mapping: {source}")
    if len(config["roots"]) != 8 or len(set(config["roots"])) != 8:
        raise ValueError("the evaluation requires eight distinct roots")
    expected_loads = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    actual_loads = tuple(float(value) for value in config["evaluation"]["loads"])
    if actual_loads != expected_loads:
        raise ValueError(f"unexpected load factors: {actual_loads}")
    return config


def project_path(value: str | Path) -> Path:
    return PROJECT_ROOT / str(value)


def data_root(config: Mapping[str, Any]) -> Path:
    return project_path(config["paths"]["data"])


def model_root(config: Mapping[str, Any]) -> Path:
    return project_path(config["paths"]["models"])


def result_root(config: Mapping[str, Any]) -> Path:
    return project_path(config["paths"]["results"])

