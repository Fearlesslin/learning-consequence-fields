from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .optimizer import SERVICE_LEVELS


EPSILON = 1e-9


def _risk(
    provider: Any,
    row: Any,
    *,
    service_level: float,
    delay_ms: float,
) -> float:
    return float(
        provider.risk(
            "evaluation",
            sample_id=str(row.sample_id),
            task_type=str(row.task_type),
            scene_variant="observed",
            goal_id=str(row.current_goal),
            service_level=float(np.clip(service_level, 0.0, 1.0)),
            delay_ms=max(float(delay_ms), 0.0),
        )
    )


def _semantic_frame(
    visible: pd.DataFrame,
    outcomes: pd.DataFrame,
    truth_provider: Any,
) -> pd.DataFrame:
    frame = visible.merge(outcomes, on="sample_id", validate="one_to_one")
    frame = frame.loc[frame["task_class"].isin(("soft_rt", "best_effort"))].copy()
    finish = pd.to_numeric(frame["finish_ms"], errors="coerce")
    delay = (finish - frame["release_ms"].astype(float)).fillna(
        frame["deadline_ms"].astype(float) * 4.0
    )
    delay = delay.where(
        frame["target_service_fraction"].astype(float).gt(EPSILON),
        frame["deadline_ms"].astype(float) * 4.0,
    )
    losses: list[float] = []
    zero_losses: list[float] = []
    values: list[float] = []
    for row, delay_ms in zip(frame.itertuples(index=False), delay, strict=True):
        losses.append(
            _risk(
                truth_provider,
                row,
                service_level=float(row.deadline_service_fraction),
                delay_ms=float(delay_ms),
            )
        )
        zero = _risk(
            truth_provider,
            row,
            service_level=0.0,
            delay_ms=float(row.deadline_ms),
        )
        full = _risk(
            truth_provider,
            row,
            service_level=1.0,
            delay_ms=0.0,
        )
        zero_losses.append(zero)
        values.append(max(zero - full, 0.0))
    frame["goal_consequence_loss"] = losses
    frame["zero_service_loss"] = zero_losses
    frame["semantic_value"] = values
    frame["response_time_ms"] = delay
    return frame


def _budget_matched_regret(frame: pd.DataFrame, truth_provider: Any) -> tuple[float, int]:
    regrets: list[float] = []
    for _, event in frame.groupby("decision_event_id", sort=True):
        event = event.sort_values("sample_id", kind="stable").reset_index(drop=True)
        observed = event["deadline_service_fraction"].to_numpy(dtype=float)
        quantized = SERVICE_LEVELS[
            np.argmin(np.abs(observed[:, None] - SERVICE_LEVELS[None, :]), axis=1)
        ]
        actual = event["execution_time_ms"].to_numpy(dtype=float)
        budget = float(quantized @ actual)
        coverage = int(np.sum(quantized >= 0.5 - EPSILON))
        vectors = np.asarray(
            list(itertools.product(SERVICE_LEVELS, repeat=len(event))), dtype=float
        )
        feasible = (vectors @ actual <= budget + 1e-7) & (
            np.sum(vectors >= 0.5 - EPSILON, axis=1) >= coverage
        )
        if not np.any(feasible):
            feasible = vectors @ actual <= budget + 1e-7
        risk_matrix = np.empty((len(event), len(SERVICE_LEVELS)), dtype=float)
        for job_index, row in enumerate(event.itertuples(index=False)):
            for level_index, level in enumerate(SERVICE_LEVELS):
                risk_matrix[job_index, level_index] = _risk(
                    truth_provider,
                    row,
                    service_level=float(level),
                    delay_ms=float(row.deadline_ms),
                )
        level_indexes = np.searchsorted(SERVICE_LEVELS, vectors)
        losses = np.take_along_axis(
            np.broadcast_to(
                risk_matrix[None, :, :], (len(vectors), *risk_matrix.shape)
            ),
            level_indexes[:, :, None],
            axis=2,
        ).squeeze(2).sum(axis=1)
        observed_indexes = np.searchsorted(SERVICE_LEVELS, quantized)
        observed_loss = float(
            risk_matrix[np.arange(len(event)), observed_indexes].sum()
        )
        oracle_loss = float(losses[feasible].min())
        scale = float(event["zero_service_loss"].sum())
        regrets.append(max(observed_loss - oracle_loss, 0.0) / max(scale, EPSILON))
    return (float(np.mean(regrets)) if regrets else 0.0, len(regrets))


def evaluate_base(
    visible: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    truth_provider: Any,
    event_diagnostics: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    frame = visible.merge(outcomes, on="sample_id", validate="one_to_one")
    target_success = frame["final_target_completed_by_deadline"].fillna(False).astype(bool)
    positive_target = frame["target_service_fraction"].astype(float).gt(EPSILON)
    optional = frame["task_class"].isin(("soft_rt", "best_effort"))
    completed_target = positive_target & frame["finish_ms"].notna()
    response = (
        frame.loc[completed_target, "finish_ms"].astype(float)
        - frame.loc[completed_target, "release_ms"].astype(float)
    )
    normalized = response / frame.loc[completed_target, "period_ms"].astype(float)
    semantic = _semantic_frame(visible, outcomes, truth_provider)
    regret, regret_events = _budget_matched_regret(semantic, truth_provider)
    high_threshold = float(semantic["semantic_value"].quantile(0.80))
    high = semantic["semantic_value"].ge(high_threshold - EPSILON)
    hard = frame["task_class"].eq("hard_rt")
    firm = frame["task_class"].eq("firm_rt")
    prefix_violations = 0
    service_anchor_ratio = 1.0
    runtime_p50 = float("nan")
    runtime_p95 = float("nan")
    runtime_p99 = float("nan")
    if event_diagnostics is not None and not event_diagnostics.empty:
        prefix_violations = int(
            (
                event_diagnostics["predicted_target_work_ms"].astype(float)
                > event_diagnostics["physical_capacity_ms"].astype(float) + 1e-7
            ).sum()
        )
        service_anchor_ratio = float(
            (
                event_diagnostics["selected_service"].astype(float)
                / event_diagnostics["s_star"].astype(float).clip(lower=EPSILON)
            ).mean()
        )
        runtime_p50 = float(event_diagnostics["decision_time_ms"].quantile(0.50))
        runtime_p95 = float(event_diagnostics["decision_time_ms"].quantile(0.95))
        runtime_p99 = float(event_diagnostics["decision_time_ms"].quantile(0.99))
    metrics: dict[str, float | int] = {
        "target_failure_rate": float(
            ((~target_success) & positive_target).sum()
            / max(int(positive_target.sum()), 1)
        ),
        "service_retention": float(
            frame["deadline_service_fraction"].astype(float).mean()
        ),
        "job_completion_rate": float(completed_target.mean()),
        "normalized_rt_p99": (
            float(normalized.quantile(0.99)) if len(normalized) else float("nan")
        ),
        "goal_conditioned_consequence_loss": float(
            semantic["goal_consequence_loss"].sum()
            / max(float(semantic["zero_service_loss"].sum()), EPSILON)
        ),
        "budget_matched_semantic_regret": regret,
        "high_consequence_service_retention": float(
            semantic.loc[high, "deadline_service_fraction"].astype(float).mean()
        ),
        "positive_service_coverage": float(
            frame.loc[optional, "target_service_fraction"]
            .astype(float)
            .ge(0.5 - EPSILON)
            .mean()
        ),
        "hard_miss_count": int((~target_success.loc[hard]).sum()),
        "firm_miss_count": int((~target_success.loc[firm]).sum()),
        "physical_prefix_violation_count": prefix_violations,
        "commitment_breach_count": int(
            frame["commitment_breach"].fillna(False).astype(bool).sum()
        ),
        "service_anchor_ratio": service_anchor_ratio,
        "semantic_decision_p50_ms": runtime_p50,
        "semantic_decision_p95_ms": runtime_p95,
        "semantic_decision_p99_ms": runtime_p99,
        "semantic_regret_event_count": int(regret_events),
    }
    return metrics, semantic
