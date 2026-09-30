from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.adaptive_replay import simulate_service_level_replay
from src.baselines.methods.paper_common import (
    base_artifact,
    prepare_jobs,
    standard_mapping,
)
from src.baselines.streams import build_stream_catalog


METHOD_NAME = "workload_aware_mc"
SOURCE_PAPER = {
    "title": (
        "Workload-Aware Scheduling of Multiple-Criticality Real-Time "
        "Applications in Vehicular Edge Computing Systems"
    ),
    "venue": "IEEE Transactions on Industrial Informatics 2023",
    "url": "https://ieeexplore.ieee.org/document/10049120",
}
CLASS_RANK = {
    "hard_rt": 0,
    "firm_rt": 1,
    "soft_rt": 2,
    "best_effort": 3,
}


def worst_fit_streams(
    predicted_utilization: dict[str, float],
    platform_cores: int,
) -> tuple[dict[str, int], list[float]]:
    loads = [0.0 for _ in range(platform_cores)]
    placement: dict[str, int] = {}
    for stream_id, utilization in sorted(
        predicted_utilization.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        core_id = min(range(platform_cores), key=lambda value: (loads[value], value))
        placement[stream_id] = core_id
        loads[core_id] += float(utilization)
    return placement, loads


def multiple_choice_service_dp(
    streams: list[dict[str, Any]],
    capacity: float,
    service_levels: list[float],
    utility_by_class: dict[str, float],
    capacity_units: int,
) -> dict[str, float]:
    if capacity_units < 1:
        raise ValueError("capacity_units must be positive")
    available = max(0, min(capacity_units, int(np.floor(capacity * capacity_units))))
    levels = sorted({float(value) for value in service_levels}, reverse=True)
    if not levels or min(levels) < 0 or max(levels) > 1:
        raise ValueError("service levels must be within [0, 1]")

    states: dict[int, tuple[float, tuple[float, ...]]] = {0: (0.0, ())}
    ordered = sorted(streams, key=lambda row: str(row["stream_id"]))
    for stream in ordered:
        utilization = float(stream["predicted_utilization"])
        utility = float(utility_by_class[str(stream["task_class"])])
        next_states: dict[int, tuple[float, tuple[float, ...]]] = {}
        for used, (value, choices) in states.items():
            for level in levels:
                cost = int(np.ceil(utilization * level * capacity_units - 1e-12))
                new_used = used + cost
                if new_used > available:
                    continue
                candidate = (value + utility * level, choices + (level,))
                incumbent = next_states.get(new_used)
                if incumbent is None or candidate > incumbent:
                    next_states[new_used] = candidate
        if not next_states:
            raise RuntimeError("multiple-choice service allocation became infeasible")
        states = next_states
    _, (_, choices) = max(
        states.items(),
        key=lambda item: (item[1][0], item[1][1], -item[0]),
    )
    return {
        str(stream["stream_id"]): float(level)
        for stream, level in zip(ordered, choices, strict=True)
    }


def build_causal_workload_decisions(
    replay_jobs: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction = method_config["prediction"]
    update_ms = int(prediction["update_interval_ms"])
    history_ms = int(prediction["trailing_window_ms"])
    if update_ms < 1 or history_ms < update_ms:
        raise ValueError("workload prediction windows are invalid")

    catalog = build_stream_catalog(full_visible)
    stream_classes = (
        full_visible[["task_type", "asset_id", "task_class"]]
        .drop_duplicates()
        .assign(
            stream_id=lambda frame: frame["task_type"].astype(str)
            + "::"
            + frame["asset_id"].astype(str)
        )
        .set_index("stream_id")["task_class"]
        .astype(str)
        .to_dict()
    )
    stream_rows = {
        str(row["stream_id"]): {
            **row,
            "task_class": stream_classes[str(row["stream_id"])],
        }
        for row in catalog.to_dict("records")
    }
    stream_ids = sorted(stream_rows)
    origin = int(np.floor(float(replay_jobs["release_ms"].min()) / update_ms) * update_ms)
    final = int(np.floor(float(replay_jobs["release_ms"].max()) / update_ms) * update_ms)
    releases = replay_jobs[["stream_id", "release_ms"]].copy()
    releases["release_ms"] = releases["release_ms"].astype(float)

    service_levels = [
        float(value) for value in method_config["allocation"]["service_levels"]
    ]
    utility = {
        str(name): float(value)
        for name, value in method_config["allocation"][
            "utility_by_task_class"
        ].items()
    }
    capacity_units = int(method_config["allocation"]["capacity_units"])
    decision_rows: list[dict[str, Any]] = []
    window_summaries: list[dict[str, Any]] = []

    for update_time in range(origin, final + update_ms, update_ms):
        history = releases[
            releases["release_ms"].ge(update_time - history_ms)
            & releases["release_ms"].lt(update_time)
        ]
        counts = history["stream_id"].value_counts()
        predicted: dict[str, float] = {}
        for stream_id in stream_ids:
            stream = stream_rows[stream_id]
            if update_time == origin:
                utilization = float(stream["utilization"])
            else:
                count = int(counts.get(stream_id, 0))
                utilization = (
                    count * float(stream["base_exec_budget_us"]) / 1000.0
                ) / history_ms
            predicted[stream_id] = float(utilization)

        placement, raw_loads = worst_fit_streams(predicted, platform_cores)
        service_by_stream = {stream_id: 1.0 for stream_id in stream_ids}
        allocated_loads = [0.0 for _ in range(platform_cores)]
        for core_id in range(platform_cores):
            assigned = [
                {
                    "stream_id": stream_id,
                    "task_class": stream_rows[stream_id]["task_class"],
                    "predicted_utilization": predicted[stream_id],
                }
                for stream_id in stream_ids
                if placement[stream_id] == core_id
            ]
            hard_load = sum(
                row["predicted_utilization"]
                for row in assigned
                if row["task_class"] == "hard_rt"
            )
            low = [row for row in assigned if row["task_class"] != "hard_rt"]
            chosen = multiple_choice_service_dp(
                low,
                capacity=max(0.0, 1.0 - hard_load),
                service_levels=service_levels,
                utility_by_class=utility,
                capacity_units=capacity_units,
            )
            service_by_stream.update(chosen)
            allocated_loads[core_id] = hard_load + sum(
                row["predicted_utilization"]
                * service_by_stream[str(row["stream_id"])]
                for row in low
            )

        for stream_id in stream_ids:
            decision_rows.append(
                {
                    "decision_time_ms": update_time,
                    "stream_id": stream_id,
                    "core_id": placement[stream_id],
                    "predicted_utilization": predicted[stream_id],
                    "target_service_fraction": service_by_stream[stream_id],
                    "history_release_count": int(counts.get(stream_id, 0)),
                    "history_upper_bound_ms": update_time,
                }
            )
        window_summaries.append(
            {
                "decision_time_ms": update_time,
                "raw_load_max": max(raw_loads),
                "allocated_load_max": max(allocated_loads),
                "degraded_streams": sum(
                    value < 1.0 - 1e-12 for value in service_by_stream.values()
                ),
            }
        )

    decisions = pd.DataFrame(decision_rows)
    jobs = replay_jobs.copy()
    jobs["decision_time_ms"] = (
        ((jobs["release_ms"].astype(float) - origin) // update_ms).astype(int)
        * update_ms
        + origin
    )
    jobs = jobs.merge(
        decisions,
        on=["decision_time_ms", "stream_id"],
        how="left",
        validate="many_to_one",
    )
    if jobs[
        ["core_id", "predicted_utilization", "target_service_fraction"]
    ].isna().any().any():
        raise RuntimeError("causal workload decisions do not cover replay jobs")
    if not jobs["release_ms"].astype(float).ge(
        jobs["history_upper_bound_ms"].astype(float)
    ).all():
        raise RuntimeError("workload predictor used a future release window")

    summaries = pd.DataFrame(window_summaries)
    diagnostics = {
        "update_interval_ms": update_ms,
        "trailing_window_ms": history_ms,
        "decision_windows": int(summaries.shape[0]),
        "future_arrivals_visible": False,
        "maximum_history_release_time_is_strictly_before_decision": True,
        "raw_partition_load_max": float(summaries["raw_load_max"].max()),
        "allocated_partition_load_max": float(
            summaries["allocated_load_max"].max()
        ),
        "windows_with_degradation": int(
            summaries["degraded_streams"].gt(0).sum()
        ),
        "mean_degraded_streams": float(summaries["degraded_streams"].mean()),
        "service_fraction_counts": {
            str(level): int(count)
            for level, count in decisions[
                "target_service_fraction"
            ].value_counts().sort_index().items()
        },
    }
    return jobs, diagnostics


def run_workload_aware_mc(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    jobs = prepare_jobs(replay_visible)
    jobs, prediction_diagnostics = build_causal_workload_decisions(
        jobs, full_visible, method_config, platform_cores
    )
    jobs["priority_tier"] = jobs["task_class"].map(CLASS_RANK).astype(int)
    outcomes = simulate_service_level_replay(
        jobs,
        platform_cores,
        partition_column="core_id",
    )
    outcomes = outcomes.merge(
        jobs[
            [
                "sample_id",
                "decision_time_ms",
                "predicted_utilization",
                "target_service_fraction",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    outcomes["scheduler_overhead_ms"] = elapsed_ms / max(len(outcomes), 1)
    artifact = base_artifact(
        method=METHOD_NAME,
        scope=method_config["implementation_scope"],
        source_paper=SOURCE_PAPER,
        full_visible=full_visible,
        platform_cores=platform_cores,
        replay_policy=method_config["replay"]["policy"],
        elapsed_ms=elapsed_ms,
    )
    artifact.update(
        {
            "prediction": prediction_diagnostics,
            "placement": method_config["placement"]["policy"],
            "allocation": {
                **method_config["allocation"],
                "hard_rt_full_service": True,
            },
            "degraded_samples": int(
                outcomes["execution_state"].eq("degraded").sum()
            ),
            "dropped_samples": int(
                outcomes["execution_state"].eq("dropped").sum()
            ),
        }
    )
    return outcomes, artifact, standard_mapping(full_visible)
