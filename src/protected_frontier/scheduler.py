from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.baselines.paper_replay import simulate_global_preemptive


def _semantic_priority(
    row: Any,
    provider: Any,
    level: float,
    laxity_band: float,
) -> tuple[float, str]:
    step = max(1.0, float(row.deadline_ms) * float(laxity_band))
    before = provider.risk(
        "dispatch",
        sample_id=str(row.sample_id),
        task_type=str(row.task_type),
        scene_variant="observed",
        goal_id=str(row.current_goal),
        service_level=float(level),
        delay_ms=0.0,
    )
    after = provider.risk(
        "dispatch",
        sample_id=str(row.sample_id),
        task_type=str(row.task_type),
        scene_variant="observed",
        goal_id=str(row.current_goal),
        service_level=float(level),
        delay_ms=step,
    )
    return -(after - before) / step, str(row.sample_id)


def build_policy_jobs(
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    target_levels: Mapping[str, float],
    provider: Any | None,
    laxity_band: float,
    priority_mode: str = "semantic",
) -> pd.DataFrame:
    jobs = visible.merge(
        hidden[["sample_id", "actual_exec_ms"]],
        on="sample_id",
        validate="one_to_one",
    ).copy()
    optional = jobs["task_class"].isin(("soft_rt", "best_effort"))
    jobs["service_fraction"] = 1.0
    jobs.loc[optional, "service_fraction"] = jobs.loc[optional, "sample_id"].map(
        lambda sample_id: float(target_levels.get(str(sample_id), 1.0))
    )
    jobs["intrinsic_exec_ms"] = jobs["actual_exec_ms"].astype(float)
    jobs["simulation_work_ms"] = (
        jobs["intrinsic_exec_ms"] * jobs["service_fraction"]
    )
    jobs["scheduler_release_ms"] = jobs["release_ms"].astype(float)
    band_width_ms = max(1.0, float(laxity_band) * 100.0)
    jobs["_deadline_band"] = np.floor(
        jobs["absolute_deadline_ms"].astype(float) / band_width_ms
    ).astype(int)
    jobs["static_priority"] = jobs["_deadline_band"] * 10_000
    optional_frame = jobs.loc[optional]
    if priority_mode == "semantic" and provider is not None:
        semantic_keys = [
            _semantic_priority(
                row,
                provider,
                float(row.service_fraction),
                float(laxity_band),
            )
            for row in optional_frame.itertuples(index=False)
        ]
    elif priority_mode == "rm":
        semantic_keys = [
            (
                int(float(row.period_ms)),
                float(row.absolute_deadline_ms),
                str(row.sample_id),
            )
            for row in optional_frame.itertuples(index=False)
        ]
    elif priority_mode == "class_edf":
        class_rank = {"soft_rt": 0, "best_effort": 1}
        semantic_keys = [
            (
                class_rank[str(row.task_class)],
                float(row.absolute_deadline_ms),
                str(row.sample_id),
            )
            for row in optional_frame.itertuples(index=False)
        ]
    else:
        semantic_keys = [
            (int(float(row.absolute_deadline_ms) * 1000), 0.0, str(row.sample_id))
            for row in optional_frame.itertuples(index=False)
        ]
    ranked: dict[str, int] = {}
    key_series = pd.Series(semantic_keys, index=optional_frame.index)
    for band, indexes in optional_frame.groupby("_deadline_band", sort=True).groups.items():
        ordered = sorted(indexes, key=lambda index: key_series.loc[index])
        for rank, index in enumerate(ordered, start=1):
            ranked[str(jobs.loc[index, "sample_id"])] = int(band) * 10_000 + rank
    jobs.loc[optional, "static_priority"] = (
        jobs.loc[optional, "sample_id"].map(ranked).astype(int)
    )
    return jobs.drop(columns="_deadline_band")


def _zero_outcomes(jobs: pd.DataFrame) -> pd.DataFrame:
    zero = jobs.loc[jobs["service_fraction"].astype(float).le(1e-12)].copy()
    if zero.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "sample_id": zero["sample_id"].astype(str),
            "execution_state": "rejected",
            "admitted": False,
            "core_id": -1,
            "first_start_ms": np.nan,
            "finish_ms": np.nan,
            "execution_time_ms": zero["intrinsic_exec_ms"].astype(float),
            "queue_wait_ms": np.nan,
            "dispatch_count": 0,
            "preemption_count": 0,
            "service_fraction": 0.0,
            "target_service_fraction": 0.0,
            "deadline_service_fraction": 0.0,
            "delivered_service_fraction": 0.0,
            "first_positive_execution_time": np.nan,
            "committed_target": 0.0,
            "target_at_deadline": 0.0,
            "maximum_target_after_first_service": 0.0,
            "minimum_target_after_first_service": 0.0,
            "post_start_downward_revision_count": 0,
            "post_start_promotion_count": 0,
            "last_downward_revision_ms_before_deadline": np.nan,
            "commitment_breach": False,
            "committed_target_completed_by_deadline": False,
            "final_target_completed_by_deadline": False,
            "interference_delay_ms": 0.0,
            "orchestration_delay_ms": 0.0,
            "scheduler_overhead_ms": 0.0,
        }
    )


def replay_policy_jobs(jobs: pd.DataFrame, channels: int) -> pd.DataFrame:
    positive = jobs.loc[jobs["simulation_work_ms"].astype(float).gt(1e-12)].copy()
    completed = simulate_global_preemptive(
        positive,
        int(channels),
        policy="fixed_priority",
    )
    result = pd.concat([completed, _zero_outcomes(jobs)], ignore_index=True, sort=False)
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def run_scheduler(
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    target_levels: Mapping[str, float],
    provider: Any,
    candidate: Any,
    config: Mapping[str, Any],
    priority_mode: str = "semantic",
) -> pd.DataFrame:
    jobs = build_policy_jobs(
        visible,
        hidden,
        target_levels=target_levels,
        provider=provider,
        laxity_band=float(candidate.laxity_band),
        priority_mode=priority_mode,
    )
    return replay_policy_jobs(jobs, int(config["platform"]["channels"]))
