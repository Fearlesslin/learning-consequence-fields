from __future__ import annotations

from collections import Counter
from time import perf_counter
from typing import Any

import pandas as pd

from src.baselines.replay import OUTCOME_COLUMNS
from src.baselines.replay_contract import (
    COMMITMENT_COLUMNS,
    CONTRACT_COLUMNS,
    contract_fields,
    initialize_target_timeline,
    record_committed_target_progress,
    record_first_positive_execution,
    record_post_start_target_revision,
    snapshot_deadline,
    snapshot_target_at_deadline,
    target_timeline_fields,
    target_work_ms,
)


EPSILON = 1e-9


def _static_key(
    row: dict[str, Any], policy: str
) -> tuple[int | float, float, float, str]:
    if policy == "edf":
        first: int | float = float(row["absolute_deadline_ms"])
    elif policy == "fixed_priority":
        first = int(row["static_priority"])
    else:
        raise ValueError(f"unknown static replay policy: {policy}")
    return (
        first,
        float(row["absolute_deadline_ms"]),
        float(row["release_ms"]),
        str(row["sample_id"]),
    )


def simulate_global_preemptive(
    jobs: pd.DataFrame,
    platform_cores: int,
    *,
    policy: str = "edf",
    scheduler_release_column: str = "scheduler_release_ms",
    work_column: str = "simulation_work_ms",
    intrinsic_column: str = "intrinsic_exec_ms",
) -> pd.DataFrame:
    required = {
        "sample_id",
        "release_ms",
        "absolute_deadline_ms",
        scheduler_release_column,
        work_column,
        intrinsic_column,
    }
    if policy == "fixed_priority":
        required.add("static_priority")
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"global replay jobs are missing columns: {missing}")
    if platform_cores < 1:
        raise ValueError("platform_cores must be at least one")
    if jobs["sample_id"].duplicated().any():
        raise ValueError("global replay received duplicate sample IDs")
    if (pd.to_numeric(jobs[work_column], errors="raise") <= 0).any():
        raise ValueError("simulation work must be positive")

    started = perf_counter()
    pending = jobs.to_dict("records")
    pending.sort(
        key=lambda row: (
            float(row[scheduler_release_column]),
            str(row["sample_id"]),
        )
    )
    for row in pending:
        row["_remaining_ms"] = float(row[work_column])
        row["_intrinsic_ms"] = float(row[intrinsic_column])
        row["_consumed_ms"] = 0.0
        row["_deadline_service_fraction"] = None
        row["_target_fraction"] = float(row.get("service_fraction", 1.0))
        expected_work = target_work_ms(
            row["_intrinsic_ms"],
            row["_target_fraction"],
            task_class=str(row.get("task_class", "non_hard")),
        )
        if abs(expected_work - float(row[work_column])) > EPSILON:
            raise ValueError("simulation work violates the common target-work contract")
        row["_first_start_ms"] = None
        row["_dispatch_count"] = 0
        row["_preemption_count"] = 0
        initialize_target_timeline(row)

    ready: dict[str, dict[str, Any]] = {}
    active: dict[int, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    index = 0
    now = float(pending[0][scheduler_release_column]) if pending else 0.0

    while index < len(pending) or ready or active:
        if not ready and not active and index < len(pending):
            now = max(now, float(pending[index][scheduler_release_column]))
        while (
            index < len(pending)
            and float(pending[index][scheduler_release_column]) <= now + EPSILON
        ):
            row = pending[index]
            ready[str(row["sample_id"])] = row
            index += 1

        candidates = list(active.values()) + list(ready.values())
        selected = sorted(
            candidates, key=lambda row: _static_key(row, policy)
        )[:platform_cores]
        selected_ids = {str(row["sample_id"]) for row in selected}

        for core_id, row in list(active.items()):
            sample_id = str(row["sample_id"])
            if sample_id not in selected_ids:
                row["_preemption_count"] += 1
                ready[sample_id] = row
                del active[core_id]

        active_ids = {str(row["sample_id"]) for row in active.values()}
        dispatches = [
            row for row in selected if str(row["sample_id"]) not in active_ids
        ]
        free_cores = [
            core_id for core_id in range(platform_cores) if core_id not in active
        ]
        for core_id, row in zip(free_cores, dispatches):
            sample_id = str(row["sample_id"])
            ready.pop(sample_id, None)
            row["_dispatch_count"] += 1
            if row["_first_start_ms"] is None:
                row["_first_start_ms"] = now
            active[core_id] = row

        if not active:
            continue

        next_release = (
            float(pending[index][scheduler_release_column])
            if index < len(pending)
            else float("inf")
        )
        next_completion = min(
            now + float(row["_remaining_ms"]) for row in active.values()
        )
        next_deadline = min(
            (
                float(row["absolute_deadline_ms"])
                for row in candidates
                if row["_deadline_service_fraction"] is None
                and float(row["absolute_deadline_ms"]) > now + EPSILON
            ),
            default=float("inf"),
        )
        event_time = min(next_release, next_completion, next_deadline)
        elapsed = event_time - now
        if elapsed < -EPSILON:
            raise RuntimeError("global replay clock moved backwards")
        for row in active.values():
            consumed_before = float(row["_consumed_ms"])
            if elapsed > EPSILON:
                record_first_positive_execution(
                    row,
                    now_ms=now,
                    target_fraction=float(row["_target_fraction"]),
                )
            row["_remaining_ms"] = max(
                0.0, float(row["_remaining_ms"]) - max(elapsed, 0.0)
            )
            row["_consumed_ms"] = float(row["_consumed_ms"]) + max(elapsed, 0.0)
            record_committed_target_progress(
                row,
                interval_start_ms=now,
                consumed_before_ms=consumed_before,
                consumed_after_ms=float(row["_consumed_ms"]),
                intrinsic_actual_ms=float(row["_intrinsic_ms"]),
            )
        now = event_time
        for row in candidates:
            snapshot_deadline(row, now)
            snapshot_target_at_deadline(
                row,
                now_ms=now,
                target_fraction=float(row["_target_fraction"]),
            )

        for core_id, row in list(active.items()):
            if float(row["_remaining_ms"]) > EPSILON:
                continue
            first_start = float(row["_first_start_ms"])
            release_ms = float(row["release_ms"])
            fields = contract_fields(
                intrinsic_actual_ms=float(row["_intrinsic_ms"]),
                consumed_ms=float(row["_consumed_ms"]),
                target_fraction=float(row["_target_fraction"]),
                deadline_fraction=row["_deadline_service_fraction"],
            )
            timeline = target_timeline_fields(
                row,
                final_target=float(row["_target_fraction"]),
                deadline_service_fraction=fields["deadline_service_fraction"],
            )
            outcome = {
                "sample_id": str(row["sample_id"]),
                "execution_state": (
                    "completed"
                    if fields["delivered_service_fraction"] >= 1.0 - EPSILON
                    else "degraded"
                ),
                "admitted": True,
                "core_id": int(core_id),
                "first_start_ms": first_start,
                "finish_ms": now,
                "execution_time_ms": float(row[intrinsic_column]),
                "queue_wait_ms": max(0.0, first_start - release_ms),
                "dispatch_count": int(row["_dispatch_count"]),
                "preemption_count": int(row["_preemption_count"]),
                "service_fraction": fields["deadline_service_fraction"],
                **fields,
                **timeline,
                "interference_delay_ms": float(
                    row.get("interference_delay_ms", 0.0)
                ),
                "orchestration_delay_ms": float(
                    row.get("orchestration_delay_ms", 0.0)
                ),
            }
            outcomes.append(outcome)
            del active[core_id]

    if len(outcomes) != len(jobs):
        raise RuntimeError(
            f"global replay completed {len(outcomes)} jobs but received {len(jobs)}"
        )
    result = pd.DataFrame(outcomes)
    elapsed_ms = (perf_counter() - started) * 1000.0
    result["scheduler_overhead_ms"] = elapsed_ms / max(len(result), 1)
    columns = OUTCOME_COLUMNS + [
        "service_fraction",
        *CONTRACT_COLUMNS,
        *COMMITMENT_COLUMNS,
        "interference_delay_ms",
        "orchestration_delay_ms",
        "scheduler_overhead_ms",
    ]
    return result[columns].sort_values(
        "sample_id", kind="stable"
    ).reset_index(drop=True)


def _mc_key(
    row: dict[str, Any],
    policy: str,
    system_hi_mode: bool,
    active_hi_streams: set[str],
    virtual_deadline_factor: float,
) -> tuple[int, float, float, str]:
    hard = str(row["task_class"]) == "hard_rt"
    stream_id = str(row["stream_id"])
    deadline = float(row["absolute_deadline_ms"])
    priority_release = float(
        row.get("priority_release_ms", row["release_ms"])
    )
    virtual_deadline = priority_release + (
        float(row["deadline_ms"]) * virtual_deadline_factor
    )
    if policy == "ca_edf":
        rank = 0 if system_hi_mode and hard else 1
    elif policy == "mc_flex":
        if hard and stream_id in active_hi_streams:
            rank = 0
        elif hard:
            rank = 1
        else:
            rank = 2
    else:
        raise ValueError(f"unknown mixed-criticality policy: {policy}")
    scheduling_deadline = virtual_deadline if hard else deadline
    return (
        rank,
        scheduling_deadline,
        priority_release,
        str(row["sample_id"]),
    )


def simulate_mixed_criticality(
    jobs: pd.DataFrame,
    platform_cores: int,
    *,
    policy: str,
    virtual_deadline_factor: float,
    degradation_fractions: dict[str, float] | None = None,
    drop_targets: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "stream_id",
        "task_class",
        "release_ms",
        "deadline_ms",
        "absolute_deadline_ms",
        "c_lo_us",
        "actual_exec_us",
    }
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"mixed-criticality jobs are missing columns: {missing}")
    if platform_cores < 1:
        raise ValueError("platform_cores must be at least one")
    if not 0 < virtual_deadline_factor <= 1:
        raise ValueError("virtual_deadline_factor must be in (0, 1]")
    if policy not in {"ca_edf", "mc_flex"}:
        raise ValueError(f"unsupported mixed-criticality policy: {policy}")

    fractions = degradation_fractions or {
        "hard_rt": 1.0,
        "firm_rt": 0.80,
        "soft_rt": 0.55,
        "best_effort": 0.30,
    }
    targets = drop_targets or {}
    for task_class, fraction in fractions.items():
        if not 0 <= float(fraction) <= 1:
            raise ValueError(
                f"degradation fraction for {task_class} must be in [0, 1]"
            )

    started = perf_counter()
    pending = jobs.to_dict("records")
    pending.sort(
        key=lambda row: (float(row["release_ms"]), str(row["sample_id"]))
    )
    for row in pending:
        intrinsic_ms = float(row["actual_exec_us"]) / 1000.0
        row["_intrinsic_ms"] = intrinsic_ms
        row["_service_target_ms"] = intrinsic_ms
        row["_consumed_ms"] = 0.0
        row["_first_start_ms"] = None
        row["_dispatch_count"] = 0
        row["_preemption_count"] = 0
        row["_overrun_detected"] = False
        row["_last_core_id"] = None
        row["_deadline_service_fraction"] = None
        row["_target_frozen"] = False
        row["_target_revision_count"] = 0
        row["_downward_revision_count"] = 0
        row["_last_downward_revision_ms_before_deadline"] = None
        row["_one_to_zero_revision_count"] = 0
        row["_one_to_half_revision_count"] = 0
        row["_target_reduction_immediate_success_count"] = 0
        row["_target_lowered_below_consumed_count"] = 0
        initialize_target_timeline(row)

    ready: dict[str, dict[str, Any]] = {}
    active: dict[int, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    active_overrun_jobs: set[str] = set()
    active_hi_stream_counts: Counter[str] = Counter()
    index = 0
    now = float(pending[0]["release_ms"]) if pending else 0.0

    def active_hi_streams() -> set[str]:
        return {
            stream_id
            for stream_id, count in active_hi_stream_counts.items()
            if count > 0
        }

    def suspended_streams() -> set[str]:
        suspended: set[str] = set()
        for stream_id in active_hi_streams():
            suspended.update(targets.get(stream_id, []))
        return suspended

    def assign_service_target(
        row: dict[str, Any], fraction: float, *, record_revision: bool
    ) -> None:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("mixed-criticality target must lie in [0,1]")
        previous = min(
            1.0,
            float(row["_service_target_ms"]) / float(row["_intrinsic_ms"]),
        )
        if record_revision and abs(previous - value) > EPSILON:
            row["_target_revision_count"] += 1
            if value < previous - EPSILON:
                row["_downward_revision_count"] += 1
                row["_last_downward_revision_ms_before_deadline"] = max(
                    0.0, float(row["absolute_deadline_ms"]) - now
                )
                if previous >= 1.0 - EPSILON and value <= EPSILON:
                    row["_one_to_zero_revision_count"] += 1
                if previous >= 1.0 - EPSILON and abs(value - 0.5) <= EPSILON:
                    row["_one_to_half_revision_count"] += 1
                consumed_fraction = min(
                    1.0,
                    float(row["_consumed_ms"]) / float(row["_intrinsic_ms"]),
                )
                if value + EPSILON < consumed_fraction:
                    row["_target_lowered_below_consumed_count"] += 1
                if (
                    consumed_fraction + EPSILON >= value
                    and consumed_fraction + EPSILON < previous
                ):
                    row["_target_reduction_immediate_success_count"] += 1
            record_post_start_target_revision(
                row,
                previous_target=previous,
                new_target=value,
                now_ms=now,
            )
        row["_service_target_ms"] = float(row["_intrinsic_ms"]) * value

    def append_outcome(
        row: dict[str, Any],
        *,
        execution_state: str,
        finish_ms: float | None,
        core_id: int | None,
    ) -> None:
        target_fraction = min(
            1.0,
            float(row["_service_target_ms"]) / float(row["_intrinsic_ms"]),
        )
        fields = contract_fields(
            intrinsic_actual_ms=float(row["_intrinsic_ms"]),
            consumed_ms=float(row["_consumed_ms"]),
            target_fraction=target_fraction,
            deadline_fraction=row["_deadline_service_fraction"],
        )
        timeline = target_timeline_fields(
            row,
            final_target=target_fraction,
            deadline_service_fraction=fields["deadline_service_fraction"],
        )
        fraction = fields["delivered_service_fraction"]
        reported_state = execution_state
        if execution_state == "completed" and fraction < 1.0 - EPSILON:
            reported_state = "degraded"
        first_start = row["_first_start_ms"]
        outcomes.append(
            {
                "sample_id": str(row["sample_id"]),
                "execution_state": reported_state,
                "admitted": True,
                "core_id": core_id,
                "first_start_ms": first_start,
                "finish_ms": finish_ms,
                "execution_time_ms": float(row["_intrinsic_ms"]),
                "queue_wait_ms": (
                    max(0.0, float(first_start) - float(row["release_ms"]))
                    if first_start is not None
                    else None
                ),
                "dispatch_count": int(row["_dispatch_count"]),
                "preemption_count": int(row["_preemption_count"]),
                "service_fraction": fields["deadline_service_fraction"],
                **fields,
                **timeline,
                "interference_delay_ms": 0.0,
                "orchestration_delay_ms": 0.0,
                "overrun_detected": bool(row["_overrun_detected"]),
                "mode_at_finish": (
                    "HI" if active_overrun_jobs else "LO"
                ),
                "target_revision_count": int(row["_target_revision_count"]),
                "downward_revision_count": int(row["_downward_revision_count"]),
                "one_to_zero_revision_count": int(
                    row["_one_to_zero_revision_count"]
                ),
                "one_to_half_revision_count": int(
                    row["_one_to_half_revision_count"]
                ),
                "target_reduction_immediate_success_count": int(
                    row["_target_reduction_immediate_success_count"]
                ),
                "target_lowered_below_consumed_count": int(
                    row["_target_lowered_below_consumed_count"]
                ),
            }
        )

    def complete_row(
        row: dict[str, Any], core_id: int | None, finish_ms: float
    ) -> None:
        sample_id = str(row["sample_id"])
        append_outcome(
            row,
            execution_state="completed",
            finish_ms=finish_ms,
            core_id=core_id,
        )
        if sample_id in active_overrun_jobs:
            active_overrun_jobs.remove(sample_id)
            stream_id = str(row["stream_id"])
            active_hi_stream_counts[stream_id] -= 1

    def drop_row(row: dict[str, Any], core_id: int | None) -> None:
        append_outcome(
            row,
            execution_state="dropped",
            finish_ms=None,
            core_id=core_id,
        )

    def apply_ca_degradation() -> None:
        if policy != "ca_edf" or not active_overrun_jobs:
            return
        for row in [*ready.values(), *active.values()]:
            task_class = str(row["task_class"])
            if task_class == "hard_rt":
                continue
            if bool(row["_target_frozen"]):
                continue
            target_fraction = min(
                float(row["_service_target_ms"]) / float(row["_intrinsic_ms"]),
                float(fractions.get(task_class, 1.0)),
            )
            assign_service_target(row, target_fraction, record_revision=True)

    def drop_suspended_jobs() -> None:
        if policy != "mc_flex":
            return
        suspended = suspended_streams()
        for sample_id, row in list(ready.items()):
            if str(row["stream_id"]) in suspended and not bool(row["_target_frozen"]):
                del ready[sample_id]
                drop_row(row, None)
        for core_id, row in list(active.items()):
            if str(row["stream_id"]) in suspended and not bool(row["_target_frozen"]):
                del active[core_id]
                drop_row(row, core_id)

    while index < len(pending) or ready or active:
        if not ready and not active and index < len(pending):
            now = max(now, float(pending[index]["release_ms"]))

        while (
            index < len(pending)
            and float(pending[index]["release_ms"]) <= now + EPSILON
        ):
            row = pending[index]
            index += 1
            if (
                policy == "mc_flex"
                and str(row["stream_id"]) in suspended_streams()
            ):
                drop_row(row, None)
                continue
            if policy == "ca_edf" and active_overrun_jobs:
                task_class = str(row["task_class"])
                if task_class != "hard_rt":
                    assign_service_target(
                        row,
                        float(fractions.get(task_class, 1.0)),
                        record_revision=False,
                    )
            ready[str(row["sample_id"])] = row

        apply_ca_degradation()
        for core_id, row in list(active.items()):
            if (
                float(row["_consumed_ms"])
                + EPSILON
                >= float(row["_service_target_ms"])
            ):
                del active[core_id]
                complete_row(row, core_id, now)
        for sample_id, row in list(ready.items()):
            if (
                float(row["_consumed_ms"])
                + EPSILON
                >= float(row["_service_target_ms"])
            ):
                del ready[sample_id]
                complete_row(row, None, now)

        if not ready and not active and index >= len(pending):
            break

        hi_streams = active_hi_streams()
        candidates = list(active.values()) + list(ready.values())
        selected = sorted(
            candidates,
            key=lambda row: _mc_key(
                row,
                policy,
                bool(active_overrun_jobs),
                hi_streams,
                virtual_deadline_factor,
            ),
        )[:platform_cores]
        selected_ids = {str(row["sample_id"]) for row in selected}

        for core_id, row in list(active.items()):
            sample_id = str(row["sample_id"])
            if sample_id not in selected_ids:
                row["_preemption_count"] += 1
                ready[sample_id] = row
                del active[core_id]

        active_ids = {str(row["sample_id"]) for row in active.values()}
        dispatches = [
            row for row in selected if str(row["sample_id"]) not in active_ids
        ]
        free_cores = [
            core_id for core_id in range(platform_cores) if core_id not in active
        ]
        for core_id, row in zip(free_cores, dispatches):
            sample_id = str(row["sample_id"])
            ready.pop(sample_id, None)
            row["_dispatch_count"] += 1
            if row["_first_start_ms"] is None:
                row["_first_start_ms"] = now
            row["_last_core_id"] = core_id
            active[core_id] = row

        if not active:
            continue

        next_release = (
            float(pending[index]["release_ms"])
            if index < len(pending)
            else float("inf")
        )
        next_completion = min(
            now
            + max(
                0.0,
                float(row["_service_target_ms"])
                - float(row["_consumed_ms"]),
            )
            for row in active.values()
        )
        next_overrun_boundary = min(
            (
                now
                + max(
                    0.0,
                    float(row["c_lo_us"]) / 1000.0
                    - float(row["_consumed_ms"]),
                )
                for row in active.values()
                if str(row["task_class"]) == "hard_rt"
                and not bool(row["_overrun_detected"])
            ),
            default=float("inf"),
        )
        candidates = [*active.values(), *ready.values()]
        next_deadline = min(
            (
                float(row["absolute_deadline_ms"])
                for row in candidates
                if row["_deadline_service_fraction"] is None
                and float(row["absolute_deadline_ms"]) > now + EPSILON
            ),
            default=float("inf"),
        )
        event_time = min(
            next_release, next_completion, next_overrun_boundary, next_deadline
        )
        elapsed = event_time - now
        if elapsed < -EPSILON:
            raise RuntimeError("mixed-criticality replay clock moved backwards")
        for row in active.values():
            consumed_before = float(row["_consumed_ms"])
            if elapsed > EPSILON:
                record_first_positive_execution(
                    row,
                    now_ms=now,
                    target_fraction=min(
                        1.0,
                        float(row["_service_target_ms"])
                        / float(row["_intrinsic_ms"]),
                    ),
                )
            row["_consumed_ms"] = min(
                float(row["_service_target_ms"]),
                float(row["_consumed_ms"]) + max(elapsed, 0.0),
            )
            record_committed_target_progress(
                row,
                interval_start_ms=now,
                consumed_before_ms=consumed_before,
                consumed_after_ms=float(row["_consumed_ms"]),
                intrinsic_actual_ms=float(row["_intrinsic_ms"]),
            )
        now = event_time

        for row in candidates:
            if (
                row["_deadline_service_fraction"] is None
                and float(row["absolute_deadline_ms"]) <= now + EPSILON
            ):
                snapshot_deadline(row, now)
                snapshot_target_at_deadline(
                    row,
                    now_ms=now,
                    target_fraction=min(
                        1.0,
                        float(row["_service_target_ms"])
                        / float(row["_intrinsic_ms"]),
                    ),
                )
                row["_target_frozen"] = True

        for core_id, row in list(active.items()):
            if (
                float(row["_consumed_ms"])
                + EPSILON
                < float(row["_service_target_ms"])
            ):
                continue
            del active[core_id]
            complete_row(row, core_id, now)

        overrun_activated = False
        for row in active.values():
            c_lo_ms = float(row["c_lo_us"]) / 1000.0
            if (
                str(row["task_class"]) == "hard_rt"
                and not bool(row["_overrun_detected"])
                and float(row["_consumed_ms"]) + EPSILON >= c_lo_ms
            ):
                row["_overrun_detected"] = True
                sample_id = str(row["sample_id"])
                active_overrun_jobs.add(sample_id)
                active_hi_stream_counts[str(row["stream_id"])] += 1
                overrun_activated = True
        if overrun_activated:
            apply_ca_degradation()
            drop_suspended_jobs()

    if len(outcomes) != len(jobs):
        raise RuntimeError(
            f"mixed-criticality replay produced {len(outcomes)} outcomes "
            f"for {len(jobs)} jobs"
        )
    result = pd.DataFrame(outcomes)
    elapsed_ms = (perf_counter() - started) * 1000.0
    result["scheduler_overhead_ms"] = elapsed_ms / max(len(result), 1)
    columns = OUTCOME_COLUMNS + [
        "service_fraction",
        *CONTRACT_COLUMNS,
        *COMMITMENT_COLUMNS,
        "interference_delay_ms",
        "orchestration_delay_ms",
        "overrun_detected",
        "mode_at_finish",
        "target_revision_count",
        "downward_revision_count",
        "one_to_zero_revision_count",
        "one_to_half_revision_count",
        "target_reduction_immediate_success_count",
        "target_lowered_below_consumed_count",
        "scheduler_overhead_ms",
    ]
    return result[columns].sort_values(
        "sample_id", kind="stable"
    ).reset_index(drop=True)
