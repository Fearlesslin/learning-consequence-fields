from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from src.baselines.methods.paper_common import (
    base_artifact,
    prepare_jobs,
    standard_mapping,
)
from src.baselines.paper_replay import simulate_mixed_criticality
from src.baselines.streams import attach_stream_ids, build_stream_catalog


METHOD_NAME = "mc_flex"
SOURCE_PAPER = {
    "title": (
        "MC_FLEX: Flexible Mixed-Criticality Real-Time Scheduling "
        "by Task-Level Mode Switch"
    ),
    "venue": "IEEE Transactions on Computers 2022",
    "doi": "10.1109/TC.2021.3111743",
    "url": "https://doi.org/10.1109/TC.2021.3111743",
}


def build_drop_targets(full_visible: pd.DataFrame) -> dict[str, list[str]]:
    catalog = build_stream_catalog(full_visible)
    stream_classes = (
        attach_stream_ids(full_visible)[["stream_id", "task_class"]]
        .drop_duplicates()
        .set_index("stream_id")["task_class"]
    )
    catalog["task_class"] = catalog["stream_id"].map(stream_classes)
    hard = catalog[catalog["task_class"].eq("hard_rt")].sort_values(
        "stream_id", kind="stable"
    )
    low = catalog[~catalog["task_class"].eq("hard_rt")].sort_values(
        ["utilization", "stream_id"], kind="stable"
    )
    if hard.empty or low.empty:
        raise ValueError("MC_FLEX requires both hard and non-hard streams")
    low_rows = low.to_dict("records")
    cursor = 0
    targets: dict[str, list[str]] = {}
    for hard_row in hard.to_dict("records"):
        required = max(
            0.0,
            float(hard_row["utilization"]) * 0.20,
        )
        selected: list[str] = []
        recovered = 0.0
        attempts = 0
        while recovered + 1e-12 < required and attempts < len(low_rows):
            candidate = low_rows[(cursor + attempts) % len(low_rows)]
            selected.append(str(candidate["stream_id"]))
            recovered += float(candidate["utilization"])
            attempts += 1
        if not selected:
            selected.append(str(low_rows[cursor % len(low_rows)]["stream_id"]))
        targets[str(hard_row["stream_id"])] = selected
        cursor = (cursor + max(attempts, 1)) % len(low_rows)
    return targets


def run_mc_flex(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    drop_targets = build_drop_targets(full_visible)
    virtual_deadline_factor = float(
        method_config["scheduling"]["virtual_deadline_factor"]
    )
    jobs = prepare_jobs(replay_visible)
    outcomes = simulate_mixed_criticality(
        jobs,
        platform_cores,
        policy="mc_flex",
        virtual_deadline_factor=virtual_deadline_factor,
        drop_targets=drop_targets,
    )

    elapsed_ms = (perf_counter() - started) * 1000.0
    outcomes["scheduler_overhead_ms"] = elapsed_ms / max(len(outcomes), 1)
    artifact = base_artifact(
        method=METHOD_NAME,
        scope=(
            "paper_inspired_task_level_mode_switch_replay_"
            "not_original_offline_schedulability_proof"
        ),
        source_paper=SOURCE_PAPER,
        full_visible=full_visible,
        platform_cores=platform_cores,
        replay_policy=(
            "edf_virtual_deadline_with_per_HI_stream_drop_and_resume_sets"
        ),
        elapsed_ms=elapsed_ms,
    )
    artifact.update(
        {
            "criticality_map": method_config["sidecars"]["criticality_map"],
            "virtual_deadline_factor": virtual_deadline_factor,
            "drop_target_selection": (
                "deterministic_low_utilization_streams_covering_each_"
                "hard_stream_C_HI_minus_C_LO_utilization"
            ),
            "drop_targets": drop_targets,
            "mode_switch_scope": "individual_hard_stream",
            "resume_rule": (
                "target streams resume when the triggering hard overrun "
                "job completes"
            ),
            "overrun_detected_samples": int(
                outcomes["overrun_detected"].astype(bool).sum()
            ),
            "dropped_samples": int(
                outcomes["execution_state"].eq("dropped").sum()
            ),
        }
    )
    return outcomes, artifact, standard_mapping(full_visible)
