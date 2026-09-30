from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from src.baselines.methods.paper_common import (
    base_artifact,
    prepare_jobs,
    standard_mapping,
)
from src.baselines.replay_contract import (
    contract_fields,
    initialize_target_timeline,
    record_committed_target_progress,
    record_first_positive_execution,
    record_post_start_target_revision,
    snapshot_deadline,
    snapshot_target_at_deadline,
    target_timeline_fields,
)

EPSILON = 1e-9
METHOD_NAME = "ca_edf"
SOURCE_PAPER = {
    "title": (
        "Criticality-Aware EDF Scheduling for Constrained-Deadline "
        "Imprecise Mixed-Criticality Systems"
    ),
    "venue": "IEEE TCAD 2024",
    "doi": "10.1109/TCAD.2023.3318512",
}


def ca_edf_delay_budget(
    *,
    now: float,
    hi_deadline: float,
    interfering_jobs: list[tuple[float, float]],
) -> float:
    eligible = sorted(
        (
            (float(deadline), float(work))
            for deadline, work in interfering_jobs
            if float(deadline) <= float(hi_deadline) + EPSILON
            and float(work) > EPSILON
        ),
        key=lambda item: item[0],
    )
    if not eligible:
        return max(float(hi_deadline) - float(now), 0.0)
    demand = 0.0
    minimum = float("inf")
    for deadline, work in eligible:
        demand += work
        minimum = min(minimum, deadline - float(now) - demand)
    return minimum


def _stream_specs(jobs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for stream_id, group in jobs.groupby("stream_id", sort=True):
        first = group.iloc[0]
        for column in ("deadline_ms", "period_ms", "c_lo_us", "c_hi_us", "task_class"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"CA-EDF stream {stream_id} has varying {column}")
        deadline = float(first["deadline_ms"])
        source_period = deadline
        specs[str(stream_id)] = {
            "stream_id": str(stream_id),
            "task_class": str(first["task_class"]),
            "deadline_ms": deadline,
            "source_period_ms": source_period,
            "reported_period_ms": float(first["period_ms"]),
            "c_lo_ms": float(first["c_lo_us"]) / 1000.0,
            "c_hi_ms": float(first["c_hi_us"]) / 1000.0,
        }
    return specs


def _partition_streams(
    specs: dict[str, dict[str, Any]], platform_cores: int
) -> tuple[dict[str, int], list[float]]:
    loads = [0.0] * int(platform_cores)
    assignment: dict[str, int] = {}
    weighted: list[tuple[float, str]] = []
    for stream_id, spec in specs.items():
        work = (
            float(spec["c_hi_ms"])
            if str(spec["task_class"]) == "hard_rt"
            else float(spec["c_lo_ms"])
        )
        weighted.append((work / float(spec["source_period_ms"]), stream_id))
    for utilization, stream_id in sorted(weighted, key=lambda item: (-item[0], item[1])):
        core_id = min(range(platform_cores), key=lambda value: (loads[value], value))
        assignment[stream_id] = core_id
        loads[core_id] += utilization
    return assignment, loads


def _initialize(row: dict[str, Any]) -> None:
    intrinsic = float(row["intrinsic_exec_ms"])
    row["_intrinsic_ms"] = intrinsic
    row["_service_target_ms"] = intrinsic
    row["_consumed_ms"] = 0.0
    row["_first_start_ms"] = None
    row["_dispatch_count"] = 0
    row["_preemption_count"] = 0
    row["_deadline_service_fraction"] = None
    row["_overrun_detected"] = False
    row["_last_delay_budget_ms"] = None
    initialize_target_timeline(row)


def _target_fraction(row: dict[str, Any]) -> float:
    return min(
        1.0,
        max(0.0, float(row["_service_target_ms"]) / float(row["_intrinsic_ms"])),
    )


def _set_target(
    row: dict[str, Any], fraction: float, *, now: float, record_revision: bool
) -> None:
    previous = _target_fraction(row)
    current = min(max(float(fraction), 0.0), 1.0)
    if record_revision and abs(previous - current) > EPSILON:
        record_post_start_target_revision(
            row,
            previous_target=previous,
            new_target=current,
            now_ms=float(now),
        )
    row["_service_target_ms"] = float(row["_intrinsic_ms"]) * current


def _edf_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(row["absolute_deadline_ms"]),
        float(row["release_ms"]),
        str(row["sample_id"]),
    )


def _projected_interference(
    *,
    now: float,
    hi_job: dict[str, Any],
    candidates: list[dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    last_release: dict[str, float],
) -> list[tuple[float, float]]:
    hi_deadline = float(hi_job["absolute_deadline_ms"])
    jobs: list[tuple[float, float]] = []
    for row in candidates:
        if str(row["sample_id"]) == str(hi_job["sample_id"]):
            continue
        deadline = float(row["absolute_deadline_ms"])
        if deadline > hi_deadline + EPSILON:
            continue
        c_lo = float(row["c_lo_us"]) / 1000.0
        remaining = max(c_lo - float(row["_consumed_ms"]), 0.0)
        if remaining > EPSILON:
            jobs.append((deadline, remaining))

    for stream_id, spec in specs.items():
        period = float(spec["source_period_ms"])
        if stream_id in last_release:
            release = float(last_release[stream_id]) + period
            while release <= float(now) + EPSILON:
                release += period
        else:
            release = float(now)
        while release < hi_deadline - EPSILON:
            deadline = release + float(spec["deadline_ms"])
            if deadline <= hi_deadline + EPSILON:
                jobs.append((deadline, float(spec["c_lo_ms"])))
            release += period
    return jobs


def _select(
    *,
    mode: str,
    candidates: list[dict[str, Any]],
    now: float,
    specs: dict[str, dict[str, Any]],
    last_release: dict[str, float],
) -> tuple[dict[str, Any], float | None, float | None]:
    if mode == "HI":
        return min(candidates, key=_edf_key), None, None

    high = sorted(
        (row for row in candidates if str(row["task_class"]) == "hard_rt"),
        key=_edf_key,
    )
    low = sorted(
        (row for row in candidates if str(row["task_class"]) != "hard_rt"),
        key=_edf_key,
    )
    if not high:
        return low[0], None, None
    if not low:
        return high[0], None, None
    high_head = high[0]
    low_head = low[0]
    if float(high_head["absolute_deadline_ms"]) <= float(
        low_head["absolute_deadline_ms"]
    ) + EPSILON:
        return high_head, None, None

    interference = _projected_interference(
        now=now,
        hi_job=high_head,
        candidates=candidates,
        specs=specs,
        last_release=last_release,
    )
    delay = ca_edf_delay_budget(
        now=now,
        hi_deadline=float(high_head["absolute_deadline_ms"]),
        interfering_jobs=interference,
    )
    high_head["_last_delay_budget_ms"] = float(delay)
    if delay <= EPSILON:
        return low_head, None, delay
    c_lo_remaining = max(
        float(high_head["c_lo_us"]) / 1000.0
        - float(high_head["_consumed_ms"]),
        0.0,
    )
    return high_head, min(delay, c_lo_remaining), delay


def _finish_outcome(
    row: dict[str, Any], *, core_id: int, now: float, mode: str
) -> dict[str, Any]:
    target = _target_fraction(row)
    fields = contract_fields(
        intrinsic_actual_ms=float(row["_intrinsic_ms"]),
        consumed_ms=float(row["_consumed_ms"]),
        target_fraction=target,
        deadline_fraction=row["_deadline_service_fraction"],
    )
    timeline = target_timeline_fields(
        row,
        final_target=target,
        deadline_service_fraction=fields["deadline_service_fraction"],
    )
    first_start = row["_first_start_ms"]
    return {
        "sample_id": str(row["sample_id"]),
        "execution_state": (
            "completed"
            if fields["delivered_service_fraction"] >= 1.0 - EPSILON
            else "degraded"
        ),
        "admitted": True,
        "core_id": int(core_id),
        "first_start_ms": first_start,
        "finish_ms": float(now),
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
        "mode_at_finish": str(mode),
        "last_ca_edf_delay_budget_ms": row["_last_delay_budget_ms"],
    }


def _simulate_core(
    core_jobs: pd.DataFrame,
    *,
    core_id: int,
    specs: dict[str, dict[str, Any]],
    degradation_fractions: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pending = core_jobs.sort_values(["release_ms", "sample_id"], kind="stable").to_dict(
        "records"
    )
    for row in pending:
        _initialize(row)

    ready: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    outcomes: list[dict[str, Any]] = []
    last_release: dict[str, float] = {}
    index = 0
    now = float(pending[0]["release_ms"]) if pending else 0.0
    mode = "LO"
    planned_until: float | None = None
    diagnostics: defaultdict[str, int] = defaultdict(int)

    def complete_if_satisfied() -> None:
        nonlocal current
        if current is not None and float(current["_consumed_ms"]) + EPSILON >= float(
            current["_service_target_ms"]
        ):
            outcomes.append(_finish_outcome(current, core_id=core_id, now=now, mode=mode))
            current = None
        for sample_id, row in list(ready.items()):
            if float(row["_consumed_ms"]) + EPSILON >= float(row["_service_target_ms"]):
                outcomes.append(_finish_outcome(row, core_id=core_id, now=now, mode=mode))
                del ready[sample_id]

    while index < len(pending) or ready or current is not None:
        if current is None and not ready:
            if mode == "HI":
                mode = "LO"
                diagnostics["mode_recoveries"] += 1
            if index < len(pending):
                now = max(now, float(pending[index]["release_ms"]))

        while index < len(pending) and float(pending[index]["release_ms"]) <= now + EPSILON:
            row = pending[index]
            index += 1
            last_release[str(row["stream_id"])] = float(row["release_ms"])
            if mode == "HI" and str(row["task_class"]) != "hard_rt":
                _set_target(
                    row,
                    degradation_fractions[str(row["task_class"])],
                    now=now,
                    record_revision=False,
                )
            ready[str(row["sample_id"])] = row

        complete_if_satisfied()
        candidates = ([current] if current is not None else []) + list(ready.values())
        if not candidates:
            continue

        selected, run_budget, delay = _select(
            mode=mode,
            candidates=candidates,
            now=now,
            specs=specs,
            last_release=last_release,
        )
        if delay is not None:
            diagnostics["delay_evaluations"] += 1
            if delay > EPSILON:
                diagnostics["lo_deadline_inversions"] += 1

        if current is None or str(current["sample_id"]) != str(selected["sample_id"]):
            if current is not None:
                current["_preemption_count"] += 1
                ready[str(current["sample_id"])] = current
            current = selected
            ready.pop(str(current["sample_id"]), None)
            current["_dispatch_count"] += 1
            if current["_first_start_ms"] is None:
                current["_first_start_ms"] = now
        planned_until = now + run_budget if run_budget is not None else None

        next_release = (
            float(pending[index]["release_ms"]) if index < len(pending) else float("inf")
        )
        next_completion = now + max(
            float(current["_service_target_ms"]) - float(current["_consumed_ms"]), 0.0
        )
        next_overrun = float("inf")
        if (
            mode == "LO"
            and str(current["task_class"]) == "hard_rt"
            and not bool(current["_overrun_detected"])
        ):
            remaining_to_lo = float(current["c_lo_us"]) / 1000.0 - float(
                current["_consumed_ms"]
            )
            if remaining_to_lo > EPSILON:
                next_overrun = now + remaining_to_lo
        all_rows = [current, *ready.values()]
        next_deadline = min(
            (
                float(row["absolute_deadline_ms"])
                for row in all_rows
                if row["_deadline_service_fraction"] is None
                and float(row["absolute_deadline_ms"]) > now + EPSILON
            ),
            default=float("inf"),
        )
        event_time = min(
            next_release,
            next_completion,
            next_overrun,
            next_deadline,
            planned_until if planned_until is not None else float("inf"),
        )
        elapsed = event_time - now
        if elapsed < -EPSILON or event_time == float("inf"):
            raise RuntimeError("source CA-EDF replay could not advance")
        elapsed = max(elapsed, 0.0)
        consumed_before = float(current["_consumed_ms"])
        if elapsed > EPSILON:
            record_first_positive_execution(
                current, now_ms=now, target_fraction=_target_fraction(current)
            )
        current["_consumed_ms"] = consumed_before + elapsed
        record_committed_target_progress(
            current,
            interval_start_ms=now,
            consumed_before_ms=consumed_before,
            consumed_after_ms=float(current["_consumed_ms"]),
            intrinsic_actual_ms=float(current["_intrinsic_ms"]),
        )
        now = event_time

        for row in all_rows:
            snapshot_deadline(row, now)
            snapshot_target_at_deadline(
                row, now_ms=now, target_fraction=_target_fraction(row)
            )

        complete_if_satisfied()
        if (
            current is not None
            and mode == "LO"
            and str(current["task_class"]) == "hard_rt"
            and not bool(current["_overrun_detected"])
            and float(current["_consumed_ms"]) + EPSILON
            >= float(current["c_lo_us"]) / 1000.0
            and float(current["_consumed_ms"]) + EPSILON
            < float(current["_service_target_ms"])
        ):
            current["_overrun_detected"] = True
            mode = "HI"
            diagnostics["mode_switches"] += 1
            for row in [current, *ready.values()]:
                if str(row["task_class"]) == "hard_rt":
                    continue
                _set_target(
                    row,
                    degradation_fractions[str(row["task_class"])],
                    now=now,
                    record_revision=True,
                )
            complete_if_satisfied()

        if mode == "HI" and current is None and not ready:
            mode = "LO"
            diagnostics["mode_recoveries"] += 1

    return outcomes, dict(diagnostics)


def run_ca_edf(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:

    started = perf_counter()
    jobs = prepare_jobs(replay_visible)
    specs = _stream_specs(jobs)
    assignment, partition_loads = _partition_streams(specs, int(platform_cores))
    jobs["core_id"] = jobs["stream_id"].map(assignment).astype(int)

    policy_path = Path(str(method_config["sidecars"]["degradation_policy"]))
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    fractions = {
        str(name): float(value)
        for name, value in payload["ca_edf"]["service_fraction_hi_mode"].items()
    }
    fractions["hard_rt"] = 1.0

    outcomes: list[dict[str, Any]] = []
    core_diagnostics: dict[str, dict[str, int]] = {}
    for core_id in range(int(platform_cores)):
        core_rows, diagnostics = _simulate_core(
            jobs.loc[jobs["core_id"].eq(core_id)].copy(),
            core_id=core_id,
            specs={
                stream_id: spec
                for stream_id, spec in specs.items()
                if assignment[stream_id] == core_id
            },
            degradation_fractions=fractions,
        )
        outcomes.extend(core_rows)
        core_diagnostics[str(core_id)] = diagnostics

    result = pd.DataFrame(outcomes).sort_values("sample_id", kind="stable").reset_index(
        drop=True
    )
    if len(result) != len(jobs):
        raise RuntimeError(
            f"CA-EDF produced {len(result)} outcomes for {len(jobs)} jobs"
        )
    elapsed_ms = (perf_counter() - started) * 1000.0
    result["scheduler_overhead_ms"] = elapsed_ms / max(len(result), 1)
    artifact = base_artifact(
        method=METHOD_NAME,
        scope="source_algorithm_partitioned_homogeneous_port_no_theorem_transfer",
        source_paper=SOURCE_PAPER,
        full_visible=full_visible,
        platform_cores=int(platform_cores),
        replay_policy="partitioned_CA_EDF_Algorithm_1_with_Eq_3_delay",
        elapsed_ms=elapsed_ms,
    )
    artifact.update(
        {
            "source_mapping": {
                "criticality": "hard_rt=HI; all other classes=LO",
                "source_period": "declared deadline (conservative implicit envelope)",
                "lo_hi_work": "configured degradation-policy service fractions",
                "multiprocessor_port": "worst-fit stream partition then Algorithm 1 per core",
            },
            "virtual_deadline_used": False,
            "adapter_degradation_strength_used": False,
            "partition_worst_case_utilization": partition_loads,
            "core_diagnostics": core_diagnostics,
            "source_schedulability_guarantee_claimed": False,
        }
    )
    return result, artifact, standard_mapping(full_visible)
