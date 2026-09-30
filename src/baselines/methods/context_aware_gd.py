from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from src.baselines.adaptive_replay import (
    close_service_dependencies,
    simulate_service_level_replay,
    validate_dependency_dag,
)
from src.baselines.methods.paper_common import (
    base_artifact,
    prepare_jobs,
    standard_mapping,
)


METHOD_NAME = "context_aware_gd"
SOURCE_PAPER = {
    "title": (
        "Context-aware Graceful Degradation for Mixed-Criticality "
        "Scheduling in Autonomous Systems"
    ),
    "venue": "IEEE TCAD 2023",
    "url": "https://eprints.whiterose.ac.uk/191184/",
}
STATE_RANK = {
    "nominal": 0,
    "attention": 1,
    "alert": 2,
    "emergency": 3,
}
CLASS_RANK = {
    "hard_rt": 0,
    "firm_rt": 1,
    "soft_rt": 2,
    "best_effort": 3,
}
OP_MODE_ALIASES = {
    "startup_ramp": "startup",
    "degradation": "disturbance",
    "recovery": "fault_recovery",
}
ALARM_LEVEL_ALIASES = {
    "normal": "none",
    "warning": "medium",
}


def _context_state(
    op_mode: str,
    alarm_level: str,
    actuation_required: bool,
    config: dict[str, Any],
) -> str:
    context = config["context"]
    normalized_mode = OP_MODE_ALIASES.get(op_mode, op_mode)
    normalized_alarm = ALARM_LEVEL_ALIASES.get(alarm_level, alarm_level)
    candidates = [
        str(context["op_mode_severity"][normalized_mode]),
        str(context["alarm_severity"][normalized_alarm]),
    ]
    rank = max(STATE_RANK[value] for value in candidates)
    if actuation_required:
        rank = max(rank, STATE_RANK["attention"])
    return next(name for name, value in STATE_RANK.items() if value == rank)


def build_context_policy(
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], list[tuple[str, str]], list[str]]:
    task_classes = (
        full_visible[["task_type", "task_class"]]
        .drop_duplicates()
        .set_index("task_type")["task_class"]
        .astype(str)
        .to_dict()
    )
    if len(task_classes) != full_visible["task_type"].nunique():
        raise ValueError("task class must be constant for each task type")
    edges = [
        (str(predecessor), str(successor))
        for predecessor, successor in method_config["functional_dependencies"]
    ]
    order = validate_dependency_dag(edges, set(task_classes))
    levels = {
        float(value)
        for value in method_config["context"]["allowed_service_levels"]
    }
    degradation_strength = float(
        method_config["context"].get("degradation_strength", 1.0)
    )
    if not 0 < degradation_strength <= 1.0:
        raise ValueError("Context-Aware GD degradation strength must be in (0, 1]")
    policy: dict[str, dict[str, float]] = {}
    for state, class_fractions in method_config["context"][
        "service_fraction"
    ].items():
        by_task = {
            task_type: (
                1.0
                if task_class == "hard_rt"
                else max(
                    levels,
                    key=lambda level: (
                        -abs(
                            level
                            - (
                                1.0
                                - degradation_strength
                                * (1.0 - float(class_fractions[task_class]))
                            )
                        ),
                        level,
                    ),
                )
            )
            for task_type, task_class in task_classes.items()
        }
        by_task = close_service_dependencies(by_task, edges, order)
        if any(value not in levels for value in by_task.values()):
            raise ValueError("dependency closure produced an unregistered service level")
        if any(
            by_task[task_type] != 1.0
            for task_type, task_class in task_classes.items()
            if task_class == "hard_rt"
        ):
            raise ValueError("Context-Aware GD must retain every hard-RT task")
        policy[str(state)] = by_task
    return policy, edges, order


def run_context_aware_gd(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    policy, edges, order = build_context_policy(full_visible, method_config)
    jobs = prepare_jobs(replay_visible)
    jobs["context_state"] = [
        _context_state(
            str(row.op_mode),
            str(row.alarm_level),
            bool(row.actuation_required),
            method_config,
        )
        for row in jobs.itertuples()
    ]
    jobs["target_service_fraction"] = [
        policy[str(row.context_state)][str(row.task_type)]
        for row in jobs.itertuples()
    ]
    jobs["priority_tier"] = [
        CLASS_RANK[str(row.task_class)] * 4
        + (3 - STATE_RANK[str(row.context_state)])
        for row in jobs.itertuples()
    ]
    outcomes = simulate_service_level_replay(jobs, platform_cores)
    context_columns = jobs[
        [
            "sample_id",
            *[
                column
                for column in (
                    "context_state",
                    "target_service_fraction",
                    "priority_tier",
                )
                if column not in outcomes.columns
            ],
        ]
    ]
    outcomes = outcomes.merge(
        context_columns, on="sample_id", how="left", validate="one_to_one"
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
            "functional_dependencies": [
                {"predecessor": predecessor, "successor": successor}
                for predecessor, successor in edges
            ],
            "dependency_topological_order": order,
            "service_policy_after_dependency_closure": policy,
            "degradation_strength": float(
                method_config["context"].get("degradation_strength", 1.0)
            ),
            "hard_rt_full_service": True,
            "context_state_counts": {
                str(name): int(count)
                for name, count in jobs["context_state"].value_counts().items()
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
