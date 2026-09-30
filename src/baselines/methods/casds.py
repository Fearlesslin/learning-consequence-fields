from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.adaptive_replay import validate_dependency_dag
from src.baselines.methods.paper_common import (
    base_artifact,
    prepare_jobs,
    standard_mapping,
)
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


METHOD_NAME = "casds"
SOURCE_PAPER = {
    "title": (
        "Context-Aware-Slack-Driven Dynamic Task Scheduling for "
        "Automotive Cyber-Physical Systems"
    ),
    "venue": "IEEE Transactions on Industrial Informatics 2026",
    "doi": "10.1109/TII.2026.3704380",
    "url": "https://doi.org/10.1109/TII.2026.3704380",
}
EPSILON = 1e-9


def build_application_model(
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
) -> dict[str, Any]:
    known_tasks = set(full_visible["task_type"].astype(str).unique())
    applications = method_config["application_model"]["applications"]
    task_to_application: dict[str, str] = {}
    downstream_rank: dict[str, int] = {}
    normalized_applications: dict[str, dict[str, Any]] = {}

    for application_name, raw in applications.items():
        name = str(application_name)
        nodes = [str(value) for value in raw["task_types"]]
        node_set = set(nodes)
        if len(nodes) != len(node_set):
            raise ValueError(f"CASDS application {name} contains duplicate task types")
        overlap = sorted(node_set & set(task_to_application))
        if overlap:
            raise ValueError(
                f"CASDS task types belong to multiple applications: {overlap}"
            )
        edges = [
            (str(predecessor), str(successor))
            for predecessor, successor in raw["edges"]
        ]
        order = validate_dependency_dag(edges, node_set)
        successors = {node: [] for node in nodes}
        for predecessor, successor in edges:
            successors[predecessor].append(successor)

        ranks: dict[str, int] = {}
        for node in reversed(order):
            ranks[node] = (
                0
                if not successors[node]
                else 1 + max(ranks[successor] for successor in successors[node])
            )
        for node in nodes:
            task_to_application[node] = name
            downstream_rank[node] = ranks[node]
        normalized_applications[name] = {
            "task_types": nodes,
            "edges": [
                {"predecessor": predecessor, "successor": successor}
                for predecessor, successor in edges
            ],
            "topological_order": order,
            "downstream_rank": ranks,
        }

    missing = sorted(known_tasks - set(task_to_application))
    unknown = sorted(set(task_to_application) - known_tasks)
    if missing or unknown:
        raise ValueError(
            "CASDS application model must cover the dataset task types exactly; "
            f"missing={missing}, unknown={unknown}"
        )
    return {
        "applications": normalized_applications,
        "task_to_application": task_to_application,
        "downstream_rank": downstream_rank,
    }


def _predicted_remaining_ms(
    row: dict[str, Any],
    execution_ewma_ms: dict[str, float],
) -> float:
    consumed_ms = float(row["_consumed_ms"])
    c_lo_ms = float(row["c_lo_us"]) / 1000.0
    c_hi_ms = float(row["c_hi_us"]) / 1000.0
    if str(row["task_class"]) == "hard_rt":
        if bool(row["_overrun_detected"]):
            predicted_total_ms = c_hi_ms
        else:
            predicted_total_ms = max(
                c_lo_ms,
                float(execution_ewma_ms.get(str(row["stream_id"]), c_lo_ms)),
            )
    else:
        predicted_total_ms = c_lo_ms
    return max(EPSILON, predicted_total_ms - consumed_ms)


def compute_context_priorities(
    candidates: list[dict[str, Any]],
    *,
    now_ms: float,
    platform_cores: int,
    execution_ewma_ms: dict[str, float],
    downstream_rank: dict[str, int],
    method_config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    if platform_cores < 1:
        raise ValueError("CASDS platform_cores must be positive")
    if not candidates:
        return {}

    criticality_scores = {
        str(name): float(value)
        for name, value in method_config["priority"]["criticality_scores"].items()
    }
    weights = np.asarray(
        [
            float(method_config["priority"]["topsis_weights"]["criticality"]),
            float(method_config["priority"]["topsis_weights"]["urgency"]),
        ],
        dtype=float,
    )
    if np.any(weights < 0.0) or not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("CASDS TOPSIS weights must be nonnegative and sum to one")

    ordered = sorted(
        candidates,
        key=lambda row: (
            float(row["absolute_deadline_ms"]),
            float(row["release_ms"]),
            str(row["sample_id"]),
        ),
    )
    predicted_by_id = {
        str(row["sample_id"]): _predicted_remaining_ms(row, execution_ewma_ms)
        for row in ordered
    }
    features: list[dict[str, float | str]] = []
    earlier_work_ms = 0.0
    for row in ordered:
        sample_id = str(row["sample_id"])
        predicted_remaining_ms = predicted_by_id[sample_id]
        predicted_interference_ms = earlier_work_ms / float(platform_cores)
        slack_ms = (
            float(row["absolute_deadline_ms"])
            - float(now_ms)
            - predicted_remaining_ms
            - predicted_interference_ms
        )
        deadline_ms = float(row["deadline_ms"])
        urgency = float(np.clip(1.0 - slack_ms / deadline_ms, 0.0, 1.0))
        task_class = str(row["task_class"])
        if task_class not in criticality_scores:
            raise ValueError(f"CASDS lacks a criticality score for {task_class}")
        task_type = str(row["task_type"])
        if task_type not in downstream_rank:
            raise ValueError(f"CASDS DAG lacks a rank for {task_type}")
        features.append(
            {
                "sample_id": sample_id,
                "criticality": criticality_scores[task_class],
                "urgency": urgency,
                "context_slack_ms": slack_ms,
                "predicted_remaining_ms": predicted_remaining_ms,
                "predicted_interference_ms": predicted_interference_ms,
                "downstream_rank": float(downstream_rank[task_type]),
            }
        )
        earlier_work_ms += predicted_remaining_ms

    matrix = np.asarray(
        [[row["criticality"], row["urgency"]] for row in features],
        dtype=float,
    )
    norms = np.linalg.norm(matrix, axis=0)
    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > EPSILON,
    )
    weighted = normalized * weights
    positive_ideal = weighted.max(axis=0)
    negative_ideal = weighted.min(axis=0)
    positive_distance = np.linalg.norm(weighted - positive_ideal, axis=1)
    negative_distance = np.linalg.norm(weighted - negative_ideal, axis=1)
    denominator = positive_distance + negative_distance
    closeness = np.divide(
        negative_distance,
        denominator,
        out=np.full_like(denominator, 0.5),
        where=denominator > EPSILON,
    )

    priorities: dict[str, dict[str, float]] = {}
    for index, row in enumerate(features):
        priorities[str(row["sample_id"])] = {
            "topsis_closeness": float(closeness[index]),
            "context_slack_ms": float(row["context_slack_ms"]),
            "predicted_remaining_ms": float(row["predicted_remaining_ms"]),
            "predicted_interference_ms": float(
                row["predicted_interference_ms"]
            ),
            "downstream_rank": float(row["downstream_rank"]),
        }
    return priorities


def _priority_key(
    row: dict[str, Any],
    priority: dict[str, float],
) -> tuple[float, float, float, float, float, str]:
    return (
        -float(priority["topsis_closeness"]),
        float(priority["context_slack_ms"]),
        -float(priority["downstream_rank"]),
        float(row["absolute_deadline_ms"]),
        float(row["release_ms"]),
        str(row["sample_id"]),
    )


def simulate_casds(
    jobs: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
    downstream_rank: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if platform_cores < 1:
        raise ValueError("CASDS platform_cores must be positive")
    pending = jobs.to_dict("records")
    pending.sort(
        key=lambda row: (float(row["release_ms"]), str(row["sample_id"]))
    )
    for row in pending:
        row["_remaining_ms"] = float(row["intrinsic_exec_ms"])
        row["_consumed_ms"] = 0.0
        row["_first_start_ms"] = None
        row["_dispatch_count"] = 0
        row["_preemption_count"] = 0
        row["_overrun_detected"] = False
        row["_last_context_slack_ms"] = None
        row["_last_topsis_closeness"] = None
        row["_intrinsic_ms"] = float(row["intrinsic_exec_ms"])
        row["_deadline_service_fraction"] = None
        initialize_target_timeline(row)

    alpha = float(method_config["history"]["ewma_alpha"])
    if not 0.0 < alpha <= 1.0:
        raise ValueError("CASDS EWMA alpha must be in (0, 1]")

    ready: dict[str, dict[str, Any]] = {}
    active: dict[int, dict[str, Any]] = {}
    outcomes: list[dict[str, Any]] = []
    execution_ewma_ms: dict[str, float] = {}
    dispatch_slacks: list[float] = []
    topsis_recomputations = 0
    topsis_multi_candidate_recomputations = 0
    nonpositive_slack_dispatches = 0
    hard_overrun_detections = 0
    ewma_updates = 0
    index = 0
    now = float(pending[0]["release_ms"]) if pending else 0.0

    while index < len(pending) or ready or active:
        if not ready and not active and index < len(pending):
            now = max(now, float(pending[index]["release_ms"]))
        while (
            index < len(pending)
            and float(pending[index]["release_ms"]) <= now + EPSILON
        ):
            row = pending[index]
            ready[str(row["sample_id"])] = row
            index += 1

        candidates = [*active.values(), *ready.values()]
        priorities = compute_context_priorities(
            candidates,
            now_ms=now,
            platform_cores=platform_cores,
            execution_ewma_ms=execution_ewma_ms,
            downstream_rank=downstream_rank,
            method_config=method_config,
        )
        if candidates:
            topsis_recomputations += 1
        if len(candidates) > 1:
            topsis_multi_candidate_recomputations += 1
        selected = sorted(
            candidates,
            key=lambda row: _priority_key(
                row, priorities[str(row["sample_id"])]
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
            priority = priorities[sample_id]
            row["_last_context_slack_ms"] = float(
                priority["context_slack_ms"]
            )
            row["_last_topsis_closeness"] = float(
                priority["topsis_closeness"]
            )
            dispatch_slacks.append(float(priority["context_slack_ms"]))
            if float(priority["context_slack_ms"]) <= 0.0:
                nonpositive_slack_dispatches += 1
            active[core_id] = row

        if not active:
            continue

        next_release = (
            float(pending[index]["release_ms"])
            if index < len(pending)
            else float("inf")
        )
        next_completion = min(
            now + float(row["_remaining_ms"]) for row in active.values()
        )
        c_lo_events: list[float] = []
        for row in active.values():
            if (
                str(row["task_class"]) == "hard_rt"
                and not bool(row["_overrun_detected"])
            ):
                c_lo_ms = float(row["c_lo_us"]) / 1000.0
                consumed_ms = float(row["_consumed_ms"])
                if consumed_ms < c_lo_ms - EPSILON:
                    c_lo_events.append(now + c_lo_ms - consumed_ms)
        next_c_lo = min(c_lo_events, default=float("inf"))
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
        event_time = min(next_release, next_completion, next_c_lo, next_deadline)
        elapsed_ms = event_time - now
        if elapsed_ms < -EPSILON:
            raise RuntimeError("CASDS replay clock moved backwards")
        elapsed_ms = max(0.0, elapsed_ms)
        for row in active.values():
            if elapsed_ms > EPSILON:
                record_first_positive_execution(
                    row, now_ms=now, target_fraction=1.0
                )
            row["_remaining_ms"] = max(
                0.0, float(row["_remaining_ms"]) - elapsed_ms
            )
            row["_consumed_ms"] = float(row["_consumed_ms"]) + elapsed_ms
        now = event_time
        for row in candidates:
            snapshot_deadline(row, now)
            snapshot_target_at_deadline(
                row, now_ms=now, target_fraction=1.0
            )

        for core_id, row in list(active.items()):
            if float(row["_remaining_ms"]) > EPSILON:
                continue
            first_start_ms = float(row["_first_start_ms"])
            fields = contract_fields(
                intrinsic_actual_ms=float(row["_intrinsic_ms"]),
                consumed_ms=float(row["_consumed_ms"]),
                target_fraction=1.0,
                deadline_fraction=row["_deadline_service_fraction"],
            )
            timeline = target_timeline_fields(
                row,
                final_target=1.0,
                deadline_service_fraction=fields["deadline_service_fraction"],
            )
            outcomes.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "execution_state": "completed",
                    "admitted": True,
                    "core_id": int(core_id),
                    "first_start_ms": first_start_ms,
                    "finish_ms": now,
                    "execution_time_ms": float(row["intrinsic_exec_ms"]),
                    "queue_wait_ms": max(
                        0.0, first_start_ms - float(row["release_ms"])
                    ),
                    "dispatch_count": int(row["_dispatch_count"]),
                    "preemption_count": int(row["_preemption_count"]),
                    "service_fraction": fields["deadline_service_fraction"],
                    **fields,
                    **timeline,
                    "interference_delay_ms": 0.0,
                    "orchestration_delay_ms": 0.0,
                    "overrun_detected": bool(row["_overrun_detected"]),
                    "last_context_slack_ms": float(
                        row["_last_context_slack_ms"]
                    ),
                    "last_topsis_closeness": float(
                        row["_last_topsis_closeness"]
                    ),
                }
            )
            stream_id = str(row["stream_id"])
            observed_ms = float(row["intrinsic_exec_ms"])
            previous_ms = float(
                execution_ewma_ms.get(
                    stream_id, float(row["c_lo_us"]) / 1000.0
                )
            )
            execution_ewma_ms[stream_id] = (
                alpha * observed_ms + (1.0 - alpha) * previous_ms
            )
            ewma_updates += 1
            del active[core_id]

        for row in active.values():
            if (
                str(row["task_class"]) == "hard_rt"
                and not bool(row["_overrun_detected"])
                and float(row["_consumed_ms"]) + EPSILON
                >= float(row["c_lo_us"]) / 1000.0
            ):
                row["_overrun_detected"] = True
                hard_overrun_detections += 1

    if len(outcomes) != len(jobs):
        raise RuntimeError(
            f"CASDS completed {len(outcomes)} jobs but received {len(jobs)}"
        )
    result = pd.DataFrame(outcomes)
    result = result.sort_values("sample_id", kind="stable").reset_index(drop=True)
    diagnostics = {
        "topsis_recomputations": int(topsis_recomputations),
        "topsis_multi_candidate_recomputations": int(
            topsis_multi_candidate_recomputations
        ),
        "nonpositive_slack_dispatches": int(nonpositive_slack_dispatches),
        "hard_overrun_detections": int(hard_overrun_detections),
        "ewma_updates": int(ewma_updates),
        "total_dispatches": int(result["dispatch_count"].sum()),
        "total_preemptions": int(result["preemption_count"].sum()),
        "dispatch_context_slack_ms": {
            "mean": float(np.mean(dispatch_slacks)),
            "p50": float(np.quantile(dispatch_slacks, 0.50)),
            "p95": float(np.quantile(dispatch_slacks, 0.95)),
            "min": float(np.min(dispatch_slacks)),
        },
        "final_execution_ewma_ms": {
            stream_id: float(value)
            for stream_id, value in sorted(execution_ewma_ms.items())
        },
    }
    return result, diagnostics


def run_casds(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    application_model = build_application_model(full_visible, method_config)
    jobs = prepare_jobs(replay_visible)
    outcomes, diagnostics = simulate_casds(
        jobs,
        method_config,
        platform_cores,
        application_model["downstream_rank"],
    )
    elapsed_ms = (perf_counter() - started) * 1000.0
    outcomes["scheduler_overhead_ms"] = elapsed_ms / max(len(outcomes), 1)
    columns = OUTCOME_COLUMNS + [
        "service_fraction",
        *CONTRACT_COLUMNS,
        *COMMITMENT_COLUMNS,
        "interference_delay_ms",
        "orchestration_delay_ms",
        "overrun_detected",
        "last_context_slack_ms",
        "last_topsis_closeness",
        "scheduler_overhead_ms",
    ]
    outcomes = outcomes[columns]

    audit = jobs[
        [
            "sample_id",
            "task_type",
            "absolute_deadline_ms",
        ]
    ].merge(
        outcomes[["sample_id", "finish_ms"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    audit["application"] = audit["task_type"].map(
        application_model["task_to_application"]
    )
    audit["deadline_satisfied"] = audit["finish_ms"].le(
        audit["absolute_deadline_ms"] + EPSILON
    )
    group_proxy: dict[str, dict[str, Any]] = {}
    for application_name, frame in audit.groupby("application", sort=True):
        group_proxy[str(application_name)] = {
            "released_jobs": int(len(frame)),
            "deadline_satisfied_jobs": int(frame["deadline_satisfied"].sum()),
            "job_deadline_satisfaction_ratio": float(
                frame["deadline_satisfied"].mean()
            ),
        }

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
            "specialization": {
                "platform": "homogeneous_parallel_execution_channels",
                "paper_faithful_claim": False,
                "dag_precedence_enforced": False,
                "release_semantics": (
                    "CSV ts_ms is already the runnable release time"
                ),
            },
            "application_model": application_model["applications"],
            "history": method_config["history"],
            "priority": method_config["priority"],
            "scheduler_diagnostics": diagnostics,
            "application_group_job_dsr_proxy": group_proxy,
            "application_group_job_dsr_proxy_is_common_metric": False,
        }
    )
    return outcomes, artifact, standard_mapping(full_visible)
