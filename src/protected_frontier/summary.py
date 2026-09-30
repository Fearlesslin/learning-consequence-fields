from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import result_root


KEYS = {"root_seed", "load", "method"}


def summarize(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = result_root(config)
    by_run = pd.read_csv(root / "metrics_by_run.csv")
    numeric = [
        column
        for column in by_run.columns
        if column not in KEYS and pd.api.types.is_numeric_dtype(by_run[column])
    ]
    summary = by_run.groupby(["load", "method"], as_index=False)[numeric].mean()
    summary.to_csv(root / "metrics_summary.csv", index=False)
    rng = np.random.default_rng(int(config["statistics"]["seed"]))
    replicates = int(config["statistics"]["bootstrap_replicates"])
    rows: list[dict[str, float | str]] = []
    for (load, method), frame in by_run.groupby(["load", "method"], sort=False):
        for metric in numeric:
            values = frame[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            draws = np.asarray(
                [
                    rng.choice(values, size=len(values), replace=True).mean()
                    for _ in range(replicates)
                ]
            )
            rows.append(
                {
                    "load": float(load),
                    "method": str(method),
                    "metric": metric,
                    "root_count": len(values),
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                }
            )
    intervals = pd.DataFrame(rows)
    intervals.to_csv(root / "confidence_intervals.csv", index=False)
    return summary, intervals
