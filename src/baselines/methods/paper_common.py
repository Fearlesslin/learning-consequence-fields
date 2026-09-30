from __future__ import annotations

from typing import Any

import pandas as pd

from src.baselines.streams import (
    attach_stream_ids,
    build_stream_catalog,
    intrinsic_exec_us,
    sample_to_stream_mapping,
)


def prepare_jobs(visible: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "ts_ms",
        "deadline_ms",
        "period_ms",
        "task_type",
        "asset_id",
        "task_class",
        "c_lo_us",
        "c_hi_us",
        "actual_exec_us",
    }
    missing = sorted(required - set(visible.columns))
    if missing:
        raise ValueError(f"paper baseline input is missing columns: {missing}")
    jobs = attach_stream_ids(visible)
    jobs["release_ms"] = jobs["ts_ms"].astype(float)
    jobs["absolute_deadline_ms"] = (
        jobs["release_ms"] + jobs["deadline_ms"].astype(float)
    )
    jobs["intrinsic_exec_us"] = intrinsic_exec_us(jobs)
    jobs["intrinsic_exec_ms"] = jobs["intrinsic_exec_us"] / 1000.0
    return jobs


def base_artifact(
    *,
    method: str,
    scope: str,
    source_paper: dict[str, str],
    full_visible: pd.DataFrame,
    platform_cores: int,
    replay_policy: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    catalog = build_stream_catalog(full_visible)
    return {
        "method": method,
        "classification": scope,
        "source_paper": source_paper,
        "platform_execution_channels": int(platform_cores),
        "stream_count": int(len(catalog)),
        "replay_policy": replay_policy,
        "arrival_source": "actual_csv_ts_ms_only",
        "skipped_slots_reconstructed": False,
        "intrinsic_execution_source": "shared_actual_exec_us",
        "overrun_visibility": "detected_only_after_consumed_C_LO",
        "hidden_labels_used_by_scheduler": False,
        "offline_scheduler_wall_time_ms": float(elapsed_ms),
        "offline_scheduler_overhead_ms_per_sample": float(elapsed_ms)
        / max(len(full_visible), 1),
    }


def standard_mapping(full_visible: pd.DataFrame) -> pd.DataFrame:
    return sample_to_stream_mapping(full_visible)
