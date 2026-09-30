from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics_base import evaluate_base


EPSILON = 1e-9


def _exact_attainment(outcomes: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=outcomes.index, dtype=float)
    if "committed_target_attainment_ms" in outcomes:
        result = pd.to_numeric(
            outcomes["committed_target_attainment_ms"], errors="coerce"
        )
    committed = pd.to_numeric(outcomes.get("committed_target"), errors="coerce")
    target = pd.to_numeric(outcomes["target_service_fraction"], errors="raise")
    delivered = pd.to_numeric(outcomes["delivered_service_fraction"], errors="raise")
    finish = pd.to_numeric(outcomes["finish_ms"], errors="coerce")
    revisions = pd.to_numeric(
        outcomes.get(
            "post_start_downward_revision_count",
            pd.Series(0, index=outcomes.index),
        ),
        errors="coerce",
    ).fillna(0)
    static_target = revisions.eq(0) & (
        committed.fillna(target).sub(target).abs().le(EPSILON)
    )
    reached = committed.fillna(0).gt(EPSILON) & delivered.add(EPSILON).ge(
        committed.fillna(np.inf)
    )
    fallback = result.isna() & static_target & reached & finish.notna()
    result.loc[fallback] = finish.loc[fallback]
    unresolved = result.isna() & reached & revisions.gt(0)
    if unresolved.any():
        raise RuntimeError("exact target-attainment time is unavailable")
    return result


def evaluate(
    visible: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    truth_provider: Any,
    event_diagnostics: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    base, semantic = evaluate_base(
        visible,
        outcomes,
        truth_provider=truth_provider,
        event_diagnostics=event_diagnostics,
    )
    frame = visible.merge(outcomes, on="sample_id", validate="one_to_one")
    attainment = _exact_attainment(outcomes)
    frame["committed_target_attainment_ms"] = frame["sample_id"].map(
        pd.Series(attainment.to_numpy(), index=outcomes["sample_id"].astype(str))
    )
    committed = pd.to_numeric(frame["committed_target"], errors="coerce")
    deadline_service = pd.to_numeric(
        frame["deadline_service_fraction"], errors="raise"
    ).clip(0.0, 1.0)
    delivered = pd.to_numeric(
        frame["delivered_service_fraction"], errors="raise"
    ).clip(0.0, 1.0)
    positive = committed.fillna(0.0).gt(EPSILON)
    timely = positive & deadline_service.add(EPSILON).ge(committed.fillna(np.inf))
    attained = positive & frame["committed_target_attainment_ms"].notna()
    response = (
        frame.loc[attained, "committed_target_attainment_ms"].astype(float)
        - frame.loc[attained, "release_ms"].astype(float)
    )
    ratio = response / frame.loc[attained, "deadline_ms"].astype(float)
    hard = frame["task_class"].eq("hard_rt")
    firm = frame["task_class"].eq("firm_rt")
    metrics: dict[str, float | int] = {
        **base,
        "target_deadline_failure_rate": float((~timely).mean()),
        "target_deadline_failure_count": int((~timely).sum()),
        "service_retention_at_deadline": float(deadline_service.mean()),
        "timely_full_service_completion_rate": float(
            deadline_service.add(EPSILON).ge(1.0).mean()
        ),
        "eventual_full_completion_rate": float(
            delivered.add(EPSILON).ge(1.0).mean()
        ),
        "mean_committed_target": float(committed.fillna(0.0).mean()),
        "p95_target_response_time_ms": (
            float(response.quantile(0.95)) if len(response) else float("nan")
        ),
        "p99_target_response_time_ms": (
            float(response.quantile(0.99)) if len(response) else float("nan")
        ),
        "target_attainment_ratio_p99": (
            float(ratio.quantile(0.99)) if len(ratio) else float("nan")
        ),
        "positive_committed_target_count": int(positive.sum()),
        "attained_positive_target_count": int(attained.sum()),
        "exact_attainment_timestamp_count": int(attainment.notna().sum()),
        "inexact_dynamic_attainment_count": 0,
        "hard_full_service_miss_count": int(
            deadline_service.loc[hard].add(EPSILON).lt(1.0).sum()
        ),
        "firm_full_service_miss_count": int(
            deadline_service.loc[firm].add(EPSILON).lt(1.0).sum()
        ),
    }
    return metrics, semantic, frame
