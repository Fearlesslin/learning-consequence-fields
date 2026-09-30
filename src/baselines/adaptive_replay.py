from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.paper_replay import simulate_global_preemptive
from src.baselines.replay import OUTCOME_COLUMNS
from src.baselines.replay_contract import COMMITMENT_COLUMNS, CONTRACT_COLUMNS


EPSILON = 1e-9


def _dropped_outcomes(jobs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in jobs.to_dict("records"):
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "execution_state": "dropped",
                "admitted": True,
                "core_id": (
                    int(row["core_id"]) if pd.notna(row.get("core_id")) else None
                ),
                "first_start_ms": None,
                "finish_ms": None,
                "execution_time_ms": float(row["intrinsic_exec_ms"]),
                "queue_wait_ms": None,
                "dispatch_count": 0,
                "preemption_count": 0,
                "service_fraction": 0.0,
                "target_service_fraction": 0.0,
                "deadline_service_fraction": 0.0,
                "delivered_service_fraction": 0.0,
                "first_positive_execution_time": None,
                "committed_target": None,
                "target_at_deadline": 0.0,
                "maximum_target_after_first_service": None,
                "minimum_target_after_first_service": None,
                "post_start_downward_revision_count": 0,
                "post_start_promotion_count": 0,
                "last_downward_revision_ms_before_deadline": None,
                "commitment_breach": False,
                "committed_target_completed_by_deadline": False,
                "final_target_completed_by_deadline": False,
                "interference_delay_ms": 0.0,
                "orchestration_delay_ms": 0.0,
                "scheduler_overhead_ms": 0.0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=OUTCOME_COLUMNS
        + [
            "service_fraction",
            *CONTRACT_COLUMNS,
            *COMMITMENT_COLUMNS,
            "interference_delay_ms",
            "orchestration_delay_ms",
            "scheduler_overhead_ms",
        ],
    )


def simulate_service_level_replay(
    jobs: pd.DataFrame,
    platform_cores: int,
    *,
    target_fraction_column: str = "target_service_fraction",
    priority_column: str = "priority_tier",
    partition_column: str | None = None,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "release_ms",
        "absolute_deadline_ms",
        "intrinsic_exec_ms",
        "c_lo_us",
        "task_class",
        target_fraction_column,
        priority_column,
    }
    if partition_column is not None:
        required.add(partition_column)
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"adaptive replay jobs are missing columns: {missing}")
    if platform_cores < 1:
        raise ValueError("platform_cores must be positive")
    fractions = pd.to_numeric(
        jobs[target_fraction_column], errors="raise"
    ).astype(float)
    if not fractions.between(0.0, 1.0).all():
        raise ValueError("target service fractions must be in [0, 1]")

    started = perf_counter()
    prepared = jobs.copy()
    prepared["service_fraction"] = fractions
    prepared["simulation_work_ms"] = (
        prepared["intrinsic_exec_ms"].astype(float) * fractions
    )
    prepared["static_priority"] = pd.to_numeric(
        prepared[priority_column], errors="raise"
    ).astype(int)
    prepared["scheduler_release_ms"] = prepared["release_ms"].astype(float)

    zero = prepared["simulation_work_ms"].le(EPSILON)
    frames: list[pd.DataFrame] = []
    if zero.any():
        frames.append(_dropped_outcomes(prepared.loc[zero]))

    runnable = prepared.loc[~zero].copy()
    if partition_column is None:
        if not runnable.empty:
            frames.append(
                simulate_global_preemptive(
                    runnable,
                    platform_cores,
                    policy="fixed_priority",
                    work_column="simulation_work_ms",
                    intrinsic_column="intrinsic_exec_ms",
                )
            )
    else:
        partition_ids = sorted(
            pd.to_numeric(
                runnable[partition_column], errors="raise"
            ).astype(int).unique()
        )
        if any(core_id < 0 or core_id >= platform_cores for core_id in partition_ids):
            raise ValueError("partition IDs must identify a platform channel")
        for core_id in partition_ids:
            partition_jobs = runnable.loc[
                pd.to_numeric(runnable[partition_column]).astype(int).eq(core_id)
            ]
            partition_result = simulate_global_preemptive(
                partition_jobs,
                1,
                policy="fixed_priority",
                work_column="simulation_work_ms",
                intrinsic_column="intrinsic_exec_ms",
            )
            dispatched = partition_result["core_id"].notna()
            partition_result.loc[dispatched, "core_id"] = core_id
            frames.append(partition_result)

    if not frames:
        raise RuntimeError("adaptive replay produced no outcomes")
    result = pd.concat(frames, ignore_index=True)
    fraction_by_id = prepared.set_index("sample_id")["service_fraction"]
    class_by_id = prepared.set_index("sample_id")["task_class"].astype(str)
    c_lo_by_id = prepared.set_index("sample_id")["c_lo_us"].astype(float) / 1000.0
    intrinsic_by_id = prepared.set_index("sample_id")["intrinsic_exec_ms"].astype(float)
    result["target_service_fraction"] = result["sample_id"].map(fraction_by_id).astype(float)
    fractional = result["delivered_service_fraction"].between(
        EPSILON, 1.0 - EPSILON, inclusive="both"
    )
    result.loc[fractional, "execution_state"] = "degraded"
    result["overrun_detected"] = (
        result["sample_id"].map(class_by_id).eq("hard_rt")
        & result["sample_id"].map(intrinsic_by_id).gt(
            result["sample_id"].map(c_lo_by_id) + EPSILON
        )
        & result["execution_state"].eq("completed")
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    result["scheduler_overhead_ms"] = elapsed_ms / max(len(result), 1)
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def validate_dependency_dag(
    edges: list[tuple[str, str]],
    known_nodes: set[str],
) -> list[str]:
    adjacency: dict[str, set[str]] = {node: set() for node in known_nodes}
    indegree = {node: 0 for node in known_nodes}
    for predecessor, successor in edges:
        if predecessor not in known_nodes or successor not in known_nodes:
            raise ValueError("functional dependency references an unknown task type")
        if successor not in adjacency[predecessor]:
            adjacency[predecessor].add(successor)
            indegree[successor] += 1
    ready = sorted(node for node, count in indegree.items() if count == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in sorted(adjacency[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(known_nodes):
        raise ValueError("functional dependency graph must be acyclic")
    return order


def close_service_dependencies(
    service_by_task: dict[str, float],
    edges: list[tuple[str, str]],
    topological_order: list[str],
) -> dict[str, float]:
    result = {name: float(value) for name, value in service_by_task.items()}
    successors: dict[str, list[str]] = {name: [] for name in result}
    for predecessor, successor in edges:
        successors[predecessor].append(successor)
    for predecessor in reversed(topological_order):
        if successors[predecessor]:
            result[predecessor] = max(
                result[predecessor],
                *(result[successor] for successor in successors[predecessor]),
            )
    return result
