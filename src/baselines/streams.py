from __future__ import annotations

from typing import Any

import pandas as pd


STREAM_KEY_FIELDS = ["task_type", "asset_id"]


def stream_id_for(task_type: str, asset_id: str) -> str:
    return f"{task_type}::{asset_id}"


def attach_stream_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["stream_id"] = [
        stream_id_for(task_type, asset_id)
        for task_type, asset_id in zip(
            result["task_type"], result["asset_id"], strict=True
        )
    ]
    return result


def intrinsic_exec_us(frame: pd.DataFrame) -> pd.Series:
    column = "actual_exec_us" if "actual_exec_us" in frame.columns else "exec_budget_us"
    values = pd.to_numeric(frame[column], errors="raise")
    if values.le(0).any():
        raise ValueError("intrinsic execution demand must be positive")
    return values.astype(int)


def _single_value(group: pd.DataFrame, column: str, stream_id: str) -> Any:
    values = group[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"stream {stream_id} has non-constant {column}: {values}")
    return values[0]


def _context_scope(group: pd.DataFrame, column: str) -> str:
    values = sorted(str(value) for value in group[column].drop_duplicates())
    return values[0] if len(values) == 1 else "mixed"


def build_stream_catalog(full_visible: pd.DataFrame) -> pd.DataFrame:
    with_streams = attach_stream_ids(full_visible)
    rows: list[dict[str, Any]] = []
    for stream_id, group in with_streams.groupby("stream_id", sort=True):
        period_ms = float(_single_value(group, "period_ms", stream_id))
        deadline_ms = float(_single_value(group, "deadline_ms", stream_id))
        exec_budget_us = int(_single_value(group, "exec_budget_us", stream_id))
        rows.append(
            {
                "stream_id": stream_id,
                "task_type": str(_single_value(group, "task_type", stream_id)),
                "asset_id": str(_single_value(group, "asset_id", stream_id)),
                "process_stage": _context_scope(group, "process_stage"),
                "sample_count": int(len(group)),
                "count": 1,
                "base_exec_budget_us": exec_budget_us,
                "effective_runtime_us": float(exec_budget_us),
                "period_us": int(round(period_ms * 1000.0)),
                "task_deadline_us": int(round(deadline_ms * 1000.0)),
                "utilization": exec_budget_us / (period_ms * 1000.0),
            }
        )
    catalog = pd.DataFrame(rows).sort_values(
        "stream_id", kind="stable"
    ).reset_index(drop=True)
    if catalog.empty:
        raise ValueError("cannot build reservations from an empty visible table")
    return catalog


def sample_to_stream_mapping(full_visible: pd.DataFrame) -> pd.DataFrame:
    mapped = attach_stream_ids(full_visible)
    return mapped[
        ["sample_id", "stream_id", "task_type", "asset_id", "process_stage"]
    ].sort_values("sample_id", kind="stable").reset_index(drop=True)
