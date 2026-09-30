from __future__ import annotations

import heapq
import math
from typing import Any

import pandas as pd

from src.baselines.replay_contract import (
    COMMITMENT_COLUMNS,
    CONTRACT_COLUMNS,
    contract_fields,
    initialize_target_timeline,
    record_committed_target_progress,
    record_first_positive_execution,
    snapshot_deadline,
    snapshot_target_at_deadline,
    target_timeline_fields,
)


EPSILON = 1e-9

OUTCOME_COLUMNS = [
    "sample_id",
    "execution_state",
    "admitted",
    "core_id",
    "first_start_ms",
    "finish_ms",
    "execution_time_ms",
    "queue_wait_ms",
    "dispatch_count",
    "preemption_count",
]


def _job_key(job: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(job["absolute_deadline_ms"]),
        float(job["release_ms"]),
        str(job["sample_id"]),
    )


def _simulate_core(core_jobs: pd.DataFrame, core_id: int) -> list[dict[str, Any]]:
    pending = core_jobs.sort_values(["release_ms", "sample_id"], kind="stable").to_dict("records")
    ready: list[tuple[tuple[float, float, str], int, dict[str, Any]]] = []
    outcomes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    index = 0
    sequence = 0
    now = float(pending[0]["release_ms"]) if pending else 0.0

    while index < len(pending) or ready or current is not None:
        while index < len(pending) and float(pending[index]["release_ms"]) <= now + EPSILON:
            job = dict(pending[index])
            job["_remaining_ms"] = float(job["execution_time_ms"])
            job["_intrinsic_ms"] = float(job["execution_time_ms"])
            job["_consumed_ms"] = 0.0
            job["_deadline_service_fraction"] = None
            job["_first_start_ms"] = None
            job["_dispatch_count"] = 0
            job["_preemption_count"] = 0
            initialize_target_timeline(job)
            heapq.heappush(ready, (_job_key(job), sequence, job))
            sequence += 1
            index += 1

        if current is not None and ready and ready[0][0] < _job_key(current):
            current["_preemption_count"] += 1
            heapq.heappush(ready, (_job_key(current), sequence, current))
            sequence += 1
            current = None

        if current is None:
            if ready:
                _, _, current = heapq.heappop(ready)
                current["_dispatch_count"] += 1
                if current["_first_start_ms"] is None:
                    current["_first_start_ms"] = now
            elif index < len(pending):
                now = max(now, float(pending[index]["release_ms"]))
                continue
            else:
                break

        next_release = (
            float(pending[index]["release_ms"])
            if index < len(pending)
            else math.inf
        )
        completion = now + float(current["_remaining_ms"])
        next_deadline = min(
            (
                float(job["absolute_deadline_ms"])
                for _, _, job in ready
                if job["_deadline_service_fraction"] is None
                and float(job["absolute_deadline_ms"]) > now + EPSILON
            ),
            default=math.inf,
        )
        if current["_deadline_service_fraction"] is None:
            deadline = float(current["absolute_deadline_ms"])
            if deadline > now + EPSILON:
                next_deadline = min(next_deadline, deadline)

        event_time = min(completion, next_release, next_deadline)
        elapsed = event_time - now
        if elapsed < -EPSILON:
            raise RuntimeError("replay clock moved backwards")
        elapsed = max(elapsed, 0.0)
        consumed_before = float(current["_consumed_ms"])
        if elapsed > EPSILON:
            record_first_positive_execution(
                current, now_ms=now, target_fraction=1.0
            )
        current["_remaining_ms"] = max(
            0.0, float(current["_remaining_ms"]) - elapsed
        )
        current["_consumed_ms"] = float(current["_consumed_ms"]) + elapsed
        record_committed_target_progress(
            current,
            interval_start_ms=now,
            consumed_before_ms=consumed_before,
            consumed_after_ms=float(current["_consumed_ms"]),
            intrinsic_actual_ms=float(current["_intrinsic_ms"]),
        )
        now = event_time
        snapshot_deadline(current, now)
        snapshot_target_at_deadline(
            current, now_ms=now, target_fraction=1.0
        )
        for _, _, job in ready:
            snapshot_deadline(job, now)
            snapshot_target_at_deadline(
                job, now_ms=now, target_fraction=1.0
            )

        if float(current["_remaining_ms"]) <= EPSILON:
            first_start = float(current["_first_start_ms"])
            fields = contract_fields(
                intrinsic_actual_ms=float(current["_intrinsic_ms"]),
                consumed_ms=float(current["_consumed_ms"]),
                target_fraction=1.0,
                deadline_fraction=current["_deadline_service_fraction"],
            )
            timeline = target_timeline_fields(
                current,
                final_target=1.0,
                deadline_service_fraction=fields["deadline_service_fraction"],
            )
            outcomes.append(
                {
                    "sample_id": str(current["sample_id"]),
                    "execution_state": "completed",
                    "admitted": True,
                    "core_id": int(core_id),
                    "first_start_ms": first_start,
                    "finish_ms": now,
                    "execution_time_ms": float(current["execution_time_ms"]),
                    "queue_wait_ms": first_start - float(current["release_ms"]),
                    "dispatch_count": int(current["_dispatch_count"]),
                    "preemption_count": int(current["_preemption_count"]),
                    **fields,
                    **timeline,
                }
            )
            current = None
            continue

    return outcomes


def simulate_partitioned_preemptive_edf(jobs: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "release_ms",
        "absolute_deadline_ms",
        "execution_time_ms",
        "core_id",
    }
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"replay jobs are missing columns: {missing}")
    if jobs.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    if (jobs["execution_time_ms"].astype(float) <= 0).any():
        raise ValueError("execution_time_ms must be positive")

    outcomes: list[dict[str, Any]] = []
    for core_id, core_jobs in jobs.groupby("core_id", sort=True):
        outcomes.extend(_simulate_core(core_jobs, int(core_id)))

    result = pd.DataFrame(
        outcomes, columns=OUTCOME_COLUMNS + CONTRACT_COLUMNS + COMMITMENT_COLUMNS
    )
    if len(result) != len(jobs):
        raise RuntimeError(f"replay completed {len(result)} jobs but received {len(jobs)}")
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)


def _fixed_priority_job_key(job: dict[str, Any]) -> tuple[int, float, str]:
    return (
        int(job["static_priority"]),
        float(job["release_ms"]),
        str(job["sample_id"]),
    )


def _simulate_fixed_priority_core(
    core_jobs: pd.DataFrame, core_id: int
) -> list[dict[str, Any]]:
    pending = core_jobs.sort_values(
        ["release_ms", "sample_id"], kind="stable"
    ).to_dict("records")
    ready: list[tuple[tuple[int, float, str], int, dict[str, Any]]] = []
    outcomes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    index = 0
    sequence = 0
    now = float(pending[0]["release_ms"]) if pending else 0.0

    while index < len(pending) or ready or current is not None:
        while (
            index < len(pending)
            and float(pending[index]["release_ms"]) <= now + EPSILON
        ):
            job = dict(pending[index])
            job["_remaining_ms"] = float(job["execution_time_ms"])
            job["_intrinsic_ms"] = float(job["execution_time_ms"])
            job["_consumed_ms"] = 0.0
            job["_deadline_service_fraction"] = None
            job["_first_start_ms"] = None
            job["_dispatch_count"] = 0
            job["_preemption_count"] = 0
            initialize_target_timeline(job)
            heapq.heappush(
                ready, (_fixed_priority_job_key(job), sequence, job)
            )
            sequence += 1
            index += 1

        if (
            current is not None
            and ready
            and ready[0][0] < _fixed_priority_job_key(current)
        ):
            current["_preemption_count"] += 1
            heapq.heappush(
                ready, (_fixed_priority_job_key(current), sequence, current)
            )
            sequence += 1
            current = None

        if current is None:
            if ready:
                _, _, current = heapq.heappop(ready)
                current["_dispatch_count"] += 1
                if current["_first_start_ms"] is None:
                    current["_first_start_ms"] = now
            elif index < len(pending):
                now = max(now, float(pending[index]["release_ms"]))
                continue
            else:
                break

        next_release = (
            float(pending[index]["release_ms"])
            if index < len(pending)
            else math.inf
        )
        completion = now + float(current["_remaining_ms"])
        next_deadline = min(
            (
                float(job["absolute_deadline_ms"])
                for _, _, job in ready
                if job["_deadline_service_fraction"] is None
                and float(job["absolute_deadline_ms"]) > now + EPSILON
            ),
            default=math.inf,
        )
        if current["_deadline_service_fraction"] is None:
            deadline = float(current["absolute_deadline_ms"])
            if deadline > now + EPSILON:
                next_deadline = min(next_deadline, deadline)

        event_time = min(completion, next_release, next_deadline)
        elapsed = event_time - now
        if elapsed < -EPSILON:
            raise RuntimeError("fixed-priority replay clock moved backwards")
        elapsed = max(elapsed, 0.0)
        consumed_before = float(current["_consumed_ms"])
        if elapsed > EPSILON:
            record_first_positive_execution(
                current, now_ms=now, target_fraction=1.0
            )
        current["_remaining_ms"] = max(
            0.0, float(current["_remaining_ms"]) - elapsed
        )
        current["_consumed_ms"] = float(current["_consumed_ms"]) + elapsed
        record_committed_target_progress(
            current,
            interval_start_ms=now,
            consumed_before_ms=consumed_before,
            consumed_after_ms=float(current["_consumed_ms"]),
            intrinsic_actual_ms=float(current["_intrinsic_ms"]),
        )
        now = event_time
        snapshot_deadline(current, now)
        snapshot_target_at_deadline(
            current, now_ms=now, target_fraction=1.0
        )
        for _, _, job in ready:
            snapshot_deadline(job, now)
            snapshot_target_at_deadline(
                job, now_ms=now, target_fraction=1.0
            )

        if float(current["_remaining_ms"]) <= EPSILON:
            first_start = float(current["_first_start_ms"])
            fields = contract_fields(
                intrinsic_actual_ms=float(current["_intrinsic_ms"]),
                consumed_ms=float(current["_consumed_ms"]),
                target_fraction=1.0,
                deadline_fraction=current["_deadline_service_fraction"],
            )
            timeline = target_timeline_fields(
                current,
                final_target=1.0,
                deadline_service_fraction=fields["deadline_service_fraction"],
            )
            outcomes.append(
                {
                    "sample_id": str(current["sample_id"]),
                    "execution_state": "completed",
                    "admitted": True,
                    "core_id": int(core_id),
                    "first_start_ms": first_start,
                    "finish_ms": now,
                    "execution_time_ms": float(current["execution_time_ms"]),
                    "queue_wait_ms": first_start - float(current["release_ms"]),
                    "dispatch_count": int(current["_dispatch_count"]),
                    "preemption_count": int(current["_preemption_count"]),
                    **fields,
                    **timeline,
                }
            )
            current = None
            continue

    return outcomes


def simulate_partitioned_preemptive_fixed_priority(
    jobs: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "release_ms",
        "absolute_deadline_ms",
        "execution_time_ms",
        "core_id",
        "static_priority",
    }
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"fixed-priority replay jobs are missing columns: {missing}")
    if jobs.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    if (jobs["execution_time_ms"].astype(float) <= 0).any():
        raise ValueError("execution_time_ms must be positive")
    priorities = pd.to_numeric(jobs["static_priority"], errors="coerce")
    if priorities.isna().any() or (priorities < 0).any():
        raise ValueError("static_priority must contain non-negative integers")
    if not priorities.eq(priorities.astype(int)).all():
        raise ValueError("static_priority must contain integer values")

    outcomes: list[dict[str, Any]] = []
    for core_id, core_jobs in jobs.groupby("core_id", sort=True):
        outcomes.extend(_simulate_fixed_priority_core(core_jobs, int(core_id)))

    result = pd.DataFrame(
        outcomes, columns=OUTCOME_COLUMNS + CONTRACT_COLUMNS + COMMITMENT_COLUMNS
    )
    if len(result) != len(jobs):
        raise RuntimeError(
            f"fixed-priority replay completed {len(result)} jobs "
            f"but received {len(jobs)}"
        )
    return result.sort_values("sample_id", kind="stable").reset_index(drop=True)
