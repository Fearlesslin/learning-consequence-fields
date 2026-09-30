from __future__ import annotations

import itertools
import math
from time import perf_counter
from typing import Any

import pandas as pd

from src.baselines.replay import OUTCOME_COLUMNS
from src.baselines.replay_contract import (
    COMMITMENT_COLUMNS,
    CONTRACT_COLUMNS,
    contract_fields,
    initialize_target_timeline,
    record_first_positive_execution,
    snapshot_deadline,
    snapshot_target_at_deadline,
    target_timeline_fields,
)
from src.baselines.streams import attach_stream_ids, stream_id_for


EPSILON = 1e-9
METHOD_NAME = "hybrid_scheduling"
SOURCE_PAPER = {
    "title": "A Hybrid Scheduling Framework for Mixed Real-Time Tasks in an Automotive System With Vehicular Network",
    "venue": "IEEE Transactions on Cloud Computing 2023",
    "doi": "10.1109/TCC.2022.3194713",
}


def _catalog(full_visible: pd.DataFrame) -> pd.DataFrame:
    frame = attach_stream_ids(full_visible)
    required = {"task_class", "period_ms", "deadline_ms", "c_lo_us", "c_hi_us"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Hybrid catalog lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    for stream_id, group in frame.groupby("stream_id", sort=True):
        def one(column: str) -> Any:
            values = group[column].drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(f"stream {stream_id} has non-constant {column}")
            return values[0]

        task_class = str(one("task_class"))
        c_us = int(one("c_hi_us") if task_class == "hard_rt" else one("c_lo_us"))
        period = float(one("period_ms"))
        rows.append(
            {
                "stream_id": str(stream_id),
                "task_type": str(one("task_type")),
                "asset_id": str(one("asset_id")),
                "task_class": task_class,
                "period_ms": period,
                "deadline_ms": float(one("deadline_ms")),
                "bound_ms": c_us / 1000.0,
                "utilization": c_us / (period * 1000.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["task_class", "period_ms", "stream_id"], kind="stable"
    ).reset_index(drop=True)


def rm_response_time_schedulable(streams: list[dict[str, Any]]) -> bool:
    ordered = sorted(
        streams,
        key=lambda row: (float(row["period_ms"]), str(row["stream_id"])),
    )
    for index, task in enumerate(ordered):
        response = float(task["bound_ms"])
        deadline = float(task["deadline_ms"])
        for _ in range(256):
            interference = sum(
                math.ceil(max(response - EPSILON, 0.0) / float(higher["period_ms"]))
                * float(higher["bound_ms"])
                for higher in ordered[:index]
            )
            updated = float(task["bound_ms"]) + interference
            if updated > deadline + EPSILON:
                return False
            if abs(updated - response) <= EPSILON:
                break
            response = updated
        else:
            return False
    return True


def _utilization(streams: list[dict[str, Any]]) -> float:
    return float(sum(float(row["utilization"]) for row in streams))


def _hard_assignments(
    hard: list[dict[str, Any]], platform_cores: int
) -> list[tuple[dict[str, int], list[list[dict[str, Any]]]]]:
    feasible = []
    for values in itertools.product(range(platform_cores), repeat=len(hard)):
        cores = [[] for _ in range(platform_cores)]
        for task, core_id in zip(hard, values, strict=True):
            cores[core_id].append(task)
        if all(rm_response_time_schedulable(core) for core in cores):
            mapping = {
                str(task["stream_id"]): int(core_id)
                for task, core_id in zip(hard, values, strict=True)
            }
            feasible.append((mapping, cores))
    if not feasible:
        raise RuntimeError("Hybrid cannot find a hard-RT-feasible partition")
    return feasible


def _pack_soft(
    soft: list[dict[str, Any]],
    initial_cores: list[list[dict[str, Any]]],
) -> tuple[dict[str, int], list[list[dict[str, Any]]]]:
    cores = [list(values) for values in initial_cores]
    placement: dict[str, int] = {}
    remaining: list[dict[str, Any]] = []
    ordered = sorted(
        soft,
        key=lambda row: (float(row["period_ms"]), str(row["stream_id"])),
    )
    for task in ordered:
        candidates = sorted(range(len(cores)), key=lambda core: (_utilization(cores[core]), core))
        chosen = next(
            (
                core
                for core in candidates
                if _utilization([*cores[core], task]) <= 0.693 + EPSILON
                and rm_response_time_schedulable([*cores[core], task])
            ),
            None,
        )
        if chosen is None:
            remaining.append(task)
        else:
            cores[chosen].append(task)
            placement[str(task["stream_id"])] = int(chosen)
    for task in remaining:
        candidates = sorted(range(len(cores)), key=lambda core: (_utilization(cores[core]), core))
        chosen = next(
            (core for core in candidates if rm_response_time_schedulable([*cores[core], task])),
            None,
        )
        if chosen is not None:
            cores[chosen].append(task)
            placement[str(task["stream_id"])] = int(chosen)
    return placement, cores


def offline_partition(
    catalog: pd.DataFrame,
    platform_cores: int,
    variant: str,
) -> tuple[dict[str, int], list[list[dict[str, Any]]]]:
    if variant not in {"ut_sd", "ut_sd_plus"}:
        raise ValueError("Hybrid variant must be ut_sd or ut_sd_plus")
    hard = catalog.loc[catalog["task_class"].eq("hard_rt")].to_dict("records")
    soft = catalog.loc[~catalog["task_class"].eq("hard_rt")].to_dict("records")
    hard_options = _hard_assignments(hard, platform_cores)
    hard_options.sort(
        key=lambda item: (
            max(_utilization(core) for core in item[1]),
            sum(_utilization(core) ** 2 for core in item[1]),
            tuple(sorted(item[0].items())),
        )
    )
    if variant == "ut_sd":
        hard_map, hard_cores = hard_options[0]
        soft_map, cores = _pack_soft(soft, hard_cores)
        return {**hard_map, **soft_map}, cores

    best: tuple[tuple[Any, ...], dict[str, int], list[list[dict[str, Any]]]] | None = None
    for hard_map, hard_cores in hard_options:
        soft_map, cores = _pack_soft(soft, hard_cores)
        mapping = {**hard_map, **soft_map}
        score = (
            -len(mapping),
            max(_utilization(core) for core in cores),
            sum(_utilization(core) ** 2 for core in cores),
            tuple(sorted(mapping.items())),
        )
        if best is None or score < best[0]:
            best = (score, mapping, cores)
    assert best is not None
    return best[1], best[2]


def _job_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (
        float(row["period_ms"]),
        float(row["release_ms"]),
        str(row["sample_id"]),
    )


def _online_core(
    row: dict[str, Any],
    *,
    now: float,
    cores: list[list[dict[str, Any]]],
    ready: dict[int, dict[str, dict[str, Any]]],
    active: dict[int, dict[str, Any]],
    variant: str,
) -> int | None:
    horizon = max(float(row["absolute_deadline_ms"]) - now, EPSILON)
    candidates = sorted(range(len(cores)), key=lambda core: (_utilization(cores[core]), core))
    for core_id in candidates:
        base_util = _utilization(cores[core_id])
        available = horizon * max(0.0, 1.0 - base_util)
        outstanding = 0.0
        for existing in [*ready[core_id].values(), *([active[core_id]] if core_id in active else [])]:
            if float(existing["absolute_deadline_ms"]) > float(row["absolute_deadline_ms"]) + EPSILON:
                continue
            consumed = float(existing["_consumed_ms"]) if variant == "ut_sd_plus" else 0.0
            outstanding += max(float(existing["bound_ms"]) - consumed, 0.0)
        if float(row["bound_ms"]) + outstanding <= available + EPSILON:
            return core_id
    return None


def simulate_hybrid(
    replay_visible: pd.DataFrame,
    catalog: pd.DataFrame,
    placement: dict[str, int],
    cores: list[list[dict[str, Any]]],
    *,
    variant: str,
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual_by_id = {
        str(row["sample_id"]): float(row["actual_exec_us"]) / 1000.0
        for row in replay_visible[["sample_id", "actual_exec_us"]].to_dict("records")
    }
    catalog_by_stream = catalog.set_index("stream_id").to_dict("index")
    forbidden = {"actual_exec_us", "actual_exec_ms", "overrun"}
    pending = replay_visible[[column for column in replay_visible.columns if column not in forbidden]].copy()
    pending["stream_id"] = [
        stream_id_for(str(task), str(asset))
        for task, asset in zip(pending["task_type"], pending["asset_id"], strict=True)
    ]
    pending["release_ms"] = pending["ts_ms"].astype(float)
    pending["absolute_deadline_ms"] = pending["release_ms"] + pending["deadline_ms"].astype(float)
    rows = pending.to_dict("records")
    rows.sort(key=lambda row: (float(row["release_ms"]), str(row["sample_id"])))
    for row in rows:
        spec = catalog_by_stream[str(row["stream_id"])]
        row["bound_ms"] = float(spec["bound_ms"])
        row["_consumed_ms"] = 0.0
        row["_remaining_ms"] = actual_by_id[str(row["sample_id"])]
        row["_intrinsic_ms"] = actual_by_id[str(row["sample_id"])]
        row["_deadline_service_fraction"] = None
        row["_first_start_ms"] = None
        row["_dispatch_count"] = 0
        row["_preemption_count"] = 0
        row["_overrun_detected"] = False
        row["_core_id"] = placement.get(str(row["stream_id"]))
        initialize_target_timeline(row)

    ready: dict[int, dict[str, dict[str, Any]]] = {core: {} for core in range(platform_cores)}
    active: dict[int, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    index = 0
    online_admitted = 0
    online_rejected = 0
    overrun_detections = 0
    now = float(rows[0]["release_ms"]) if rows else 0.0

    def append(row: dict[str, Any], state: str, core_id: int | None, finish: float | None) -> None:
        actual = actual_by_id[str(row["sample_id"])]
        fields = contract_fields(
            intrinsic_actual_ms=actual,
            consumed_ms=float(row["_consumed_ms"]),
            target_fraction=1.0 if state != "unadmitted" else 0.0,
            deadline_fraction=row["_deadline_service_fraction"],
        )
        target = 1.0 if state != "unadmitted" else 0.0
        timeline = target_timeline_fields(
            row,
            final_target=target,
            deadline_service_fraction=fields["deadline_service_fraction"],
        )
        first = row["_first_start_ms"]
        outcomes.append(
            {
                "sample_id": str(row["sample_id"]),
                "execution_state": state,
                "admitted": state != "unadmitted",
                "core_id": core_id,
                "first_start_ms": first,
                "finish_ms": finish,
                "execution_time_ms": actual,
                "queue_wait_ms": max(0.0, float(first) - float(row["release_ms"])) if first is not None else None,
                "dispatch_count": int(row["_dispatch_count"]),
                "preemption_count": int(row["_preemption_count"]),
                "service_fraction": fields["deadline_service_fraction"],
                **fields,
                **timeline,
                "overrun_detected": bool(row["_overrun_detected"]),
                "interference_delay_ms": 0.0,
                "orchestration_delay_ms": 0.0,
            }
        )

    while index < len(rows) or any(ready.values()) or active:
        if not active and not any(ready.values()) and index < len(rows):
            now = max(now, float(rows[index]["release_ms"]))
        while index < len(rows) and float(rows[index]["release_ms"]) <= now + EPSILON:
            row = rows[index]
            core_id = row["_core_id"]
            if core_id is None:
                core_id = _online_core(
                    row,
                    now=now,
                    cores=cores,
                    ready=ready,
                    active=active,
                    variant=variant,
                )
                if core_id is None:
                    append(row, "unadmitted", None, None)
                    online_rejected += 1
                    index += 1
                    continue
                row["_core_id"] = core_id
                online_admitted += 1
            ready[int(core_id)][str(row["sample_id"])] = row
            index += 1

        for core_id in range(platform_cores):
            candidates = [*( [active[core_id]] if core_id in active else []), *ready[core_id].values()]
            if not candidates:
                continue
            selected = min(candidates, key=_job_key)
            selected_id = str(selected["sample_id"])
            if core_id in active and str(active[core_id]["sample_id"]) != selected_id:
                old = active.pop(core_id)
                old["_preemption_count"] += 1
                ready[core_id][str(old["sample_id"])] = old
            if core_id not in active:
                ready[core_id].pop(selected_id, None)
                selected["_dispatch_count"] += 1
                if selected["_first_start_ms"] is None:
                    selected["_first_start_ms"] = now
                active[core_id] = selected

        if not active:
            continue
        next_release = float(rows[index]["release_ms"]) if index < len(rows) else float("inf")
        next_completion = min(now + float(row["_remaining_ms"]) for row in active.values())
        next_c_lo = min(
            (
                now + max(float(row["c_lo_us"]) / 1000.0 - float(row["_consumed_ms"]), 0.0)
                for row in active.values()
                if str(row["task_class"]) == "hard_rt"
                and not bool(row["_overrun_detected"])
                and float(row["_consumed_ms"]) < float(row["c_lo_us"]) / 1000.0 - EPSILON
            ),
            default=float("inf"),
        )
        unfinished = [*active.values(), *(row for group in ready.values() for row in group.values())]
        next_deadline = min(
            (
                float(row["absolute_deadline_ms"])
                for row in unfinished
                if row["_deadline_service_fraction"] is None
                and float(row["absolute_deadline_ms"]) > now + EPSILON
            ),
            default=float("inf"),
        )
        event_time = min(next_release, next_completion, next_c_lo, next_deadline)
        elapsed = max(event_time - now, 0.0)
        for row in active.values():
            if elapsed > EPSILON:
                record_first_positive_execution(
                    row, now_ms=now, target_fraction=1.0
                )
            row["_remaining_ms"] = max(0.0, float(row["_remaining_ms"]) - elapsed)
            row["_consumed_ms"] = float(row["_consumed_ms"]) + elapsed
        now = event_time
        for row in unfinished:
            snapshot_deadline(row, now)
            snapshot_target_at_deadline(
                row, now_ms=now, target_fraction=1.0
            )
        for core_id, row in list(active.items()):
            if float(row["_remaining_ms"]) <= EPSILON:
                append(row, "completed", core_id, now)
                del active[core_id]
        for row in active.values():
            if (
                str(row["task_class"]) == "hard_rt"
                and not bool(row["_overrun_detected"])
                and float(row["_consumed_ms"]) >= float(row["c_lo_us"]) / 1000.0 - EPSILON
            ):
                row["_overrun_detected"] = True
                overrun_detections += 1

    result = pd.DataFrame(outcomes).sort_values("sample_id", kind="stable").reset_index(drop=True)
    columns = OUTCOME_COLUMNS + [
        "service_fraction",
        *CONTRACT_COLUMNS,
        *COMMITMENT_COLUMNS,
        "overrun_detected",
        "interference_delay_ms",
        "orchestration_delay_ms",
    ]
    diagnostics = {
        "offline_assigned_streams": int(len(placement)),
        "offline_unassigned_streams": int(len(catalog) - len(placement)),
        "online_admitted_jobs": int(online_admitted),
        "online_rejected_jobs": int(online_rejected),
        "hard_overrun_detections": int(overrun_detections),
    }
    return result[columns], diagnostics


def run_hybrid_scheduling(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    variant = str(method_config["assignment_variant"])
    catalog = _catalog(full_visible)
    placement, cores = offline_partition(catalog, platform_cores, variant)
    outcomes, diagnostics = simulate_hybrid(
        replay_visible,
        catalog,
        placement,
        cores,
        variant=variant,
        platform_cores=platform_cores,
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    outcomes["scheduler_overhead_ms"] = elapsed_ms / max(len(outcomes), 1)
    artifact = catalog.copy()
    artifact["core_id"] = artifact["stream_id"].map(placement)
    artifact["offline_assigned"] = artifact["core_id"].notna()
    diagnostics.update(
        {
            "method": METHOD_NAME,
            "candidate": f"hybrid_{variant}",
            "source_paper": SOURCE_PAPER,
            "adaptation": "homogeneous_two_channel_local_only_no_edge_or_cloud_offload",
            "local_scheduler": "partitioned_preemptive_rate_monotonic",
            "offline_test": "liu_layland_0.693_then_fixed_priority_demand_supply_recurrence",
            "online_test": "causal_local_constraint_20_specialization",
            "hidden_labels_used_by_scheduler": False,
            "actual_execution_visible_before_c_lo": False,
            "scheduler_wall_time_ms": elapsed_ms,
        }
    )
    return outcomes, diagnostics, artifact
