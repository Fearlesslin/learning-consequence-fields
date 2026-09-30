from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.adaptive_replay import simulate_service_level_replay
from src.baselines.methods.paper_common import (
    base_artifact,
    prepare_jobs,
    standard_mapping,
)


METHOD_NAME = "slack_time_management"
SOURCE_PAPER = {
    "title": "Slack-Time Management for Energy-Reliability Constrained Scheduling",
    "venue": "IEEE Transactions on Computers 2025",
    "scope": "homogeneous service-scheduling adaptation",
}


def run_slack_time_management(
    replay_visible: pd.DataFrame,
    full_visible: pd.DataFrame,
    method_config: dict[str, Any],
    platform_cores: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    jobs = prepare_jobs(replay_visible)
    policy = method_config["slack"]
    base_service = float(policy["base_non_hard_service"])
    reclaim_threshold = float(policy["reclaim_threshold"])
    if base_service not in {0.5, 0.75, 1.0}:
        raise ValueError("base_non_hard_service must be 0.5, 0.75, or 1.0")
    if not 0.0 <= reclaim_threshold <= 1.0:
        raise ValueError("reclaim_threshold must lie in [0, 1]")

    declared = jobs["c_lo_us"].astype(float) / 1000.0
    relative_slack = (
        (jobs["deadline_ms"].astype(float) - declared)
        / jobs["deadline_ms"].astype(float).clip(lower=1e-9)
    ).clip(0.0, 1.0)
    hard = jobs["task_class"].eq("hard_rt")
    firm = jobs["task_class"].eq("firm_rt")
    reclaimable = relative_slack.ge(reclaim_threshold)
    jobs["target_service_fraction"] = np.where(
        hard | firm | reclaimable,
        1.0,
        base_service,
    )
    class_rank = jobs["task_class"].map(
        {"hard_rt": 0, "firm_rt": 1, "soft_rt": 2, "best_effort": 3}
    )
    jobs["priority_tier"] = class_rank.astype(int)
    outcomes = simulate_service_level_replay(jobs, platform_cores)

    elapsed_ms = (perf_counter() - started) * 1000.0
    outcomes["scheduler_overhead_ms"] = elapsed_ms / max(len(outcomes), 1)
    artifact = base_artifact(
        method=METHOD_NAME,
        scope="paper_inspired_homogeneous_slack_reclaim_adaptation",
        source_paper=SOURCE_PAPER,
        full_visible=full_visible,
        platform_cores=platform_cores,
        replay_policy="class_safe_edf_with_static_slack_service_reclaim",
        elapsed_ms=elapsed_ms,
    )
    artifact.update(
        {
            "base_non_hard_service": base_service,
            "reclaim_threshold": reclaim_threshold,
            "dvfs_or_checkpointing_reproduced": False,
            "future_arrivals_visible": False,
            "hidden_labels_used": False,
            "reclaimed_full_service_jobs": int((~hard & ~firm & reclaimable).sum()),
        }
    )
    return outcomes, artifact, standard_mapping(full_visible)
