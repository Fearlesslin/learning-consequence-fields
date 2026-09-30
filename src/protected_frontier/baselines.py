from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import yaml

from src.baselines.methods import METHOD_RUNNERS
from src.baselines.methods.paper_common import prepare_jobs
from src.baselines.paper_replay import simulate_global_preemptive

from .config import project_path


DISPLAY_NAMES = {
    "mc_flex": "MC_FLEX",
    "workload_aware_mc": "Workload-Aware MC",
    "ca_edf": "CA-EDF",
    "context_aware_gd": "Context-Aware GD",
    "slack_time_management": "Slack-Time Management",
    "casds": "CASDS",
    "hybrid_scheduling": "Hybrid Scheduling",
    "edf_full": "EDF-Full",
    "protected_frontier": "Protected-Frontier Scheduler",
    "no_reserve": "A1 w/o Uncertainty Reserve",
    "no_goal": "A2 w/o Goal Conditioning",
    "scalar_priority": "A3 Scalar Priority",
    "static_type": "Static Task-Type Consequence",
}


def load_method_configs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = project_path(config["paths"]["methods"])
    methods = yaml.safe_load(path.read_text(encoding="utf-8"))
    for spec in methods.values():
        for key, value in spec.get("sidecars", {}).items():
            spec["sidecars"][key] = str(project_path(value))
    return {str(name): dict(spec) for name, spec in methods.items()}


def replay_input(visible: pd.DataFrame, hidden: pd.DataFrame) -> pd.DataFrame:
    return visible.merge(
        hidden[["sample_id", "actual_exec_us"]],
        on="sample_id",
        validate="one_to_one",
    ).copy()


def _run_edf_full(frame: pd.DataFrame, channels: int) -> pd.DataFrame:
    jobs = prepare_jobs(frame)
    jobs["scheduler_release_ms"] = jobs["release_ms"]
    jobs["service_fraction"] = 1.0
    jobs["simulation_work_ms"] = jobs["intrinsic_exec_ms"]
    return simulate_global_preemptive(jobs, int(channels), policy="edf")


def run_baseline(
    method_id: str,
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    method_config: Mapping[str, Any],
    *,
    channels: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = replay_input(visible, hidden)
    if method_id == "edf_full":
        return _run_edf_full(frame, channels), {
            "method": method_id,
            "classification": "global_preemptive_edf",
            "hidden_labels_used": False,
        }
    runner = METHOD_RUNNERS[method_id]
    outcome, artifact, _ = runner(
        frame,
        frame,
        dict(method_config),
        int(channels),
    )
    if "target_service_fraction" not in outcome.columns:
        left = "target_service_fraction_x"
        right = "target_service_fraction_y"
        if left in outcome.columns and right in outcome.columns:
            if not outcome[left].astype(float).equals(outcome[right].astype(float)):
                raise RuntimeError(
                    f"{method_id} returned conflicting target-service columns"
                )
            outcome = outcome.rename(columns={left: "target_service_fraction"}).drop(
                columns=right
            )
    return outcome, artifact
