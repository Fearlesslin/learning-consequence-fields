from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


SERVICE_LEVELS = np.asarray((0.0, 0.5, 0.75, 1.0), dtype=float)


class RiskProvider(Protocol):
    def risk(self, method_id: str, **kwargs: Any) -> float: ...


@dataclass(frozen=True)
class SchedulerCandidate:
    candidate_id: str
    epsilon_service: float
    reserve_quantile: float
    laxity_band: float


@dataclass(frozen=True)
class DemandProfile:
    median_ratio: dict[str, float]
    q90_ratio: dict[str, float]
    q95_ratio: dict[str, float]

    def ratio(self, task_type: str, quantile: float) -> float:
        table = self.q95_ratio if float(quantile) >= 0.95 else self.q90_ratio
        return float(table[str(task_type)])


@dataclass(frozen=True)
class EventSolution:
    event_id: str
    selected_levels: dict[str, float]
    k_star: int
    s_star: float
    selected_service: float
    physical_capacity_ms: float
    predicted_target_work_ms: float
    reserve_ms: float
    same_coverage_vectors: int
    envelope_vectors: int
    objective_value: float
    decision_time_ms: float


def load_demand_profile(path: Path) -> DemandProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DemandProfile(
        median_ratio={str(k): float(v) for k, v in payload["median_ratio"].items()},
        q90_ratio={str(k): float(v) for k, v in payload["q90_ratio"].items()},
        q95_ratio={str(k): float(v) for k, v in payload["q95_ratio"].items()},
    )


def candidate_from_config(config: Mapping[str, Any]) -> SchedulerCandidate:
    scheduler = config["scheduler"]
    return SchedulerCandidate(
        candidate_id="default",
        epsilon_service=float(scheduler["epsilon_service"]),
        reserve_quantile=float(scheduler["reserve_quantile"]),
        laxity_band=float(scheduler["laxity_band"]),
    )


def residual_capacity_ms(config: Mapping[str, Any], horizon_ms: float) -> float:
    mandatory_utilization = sum(
        float(row["c_lo_ms"]) / float(row["period_ms"])
        for row in config["task_catalog"].values()
        if str(row["class"]) in {"hard_rt", "firm_rt"}
    )
    return max(
        0.0,
        float(horizon_ms)
        * (int(config["platform"]["channels"]) - mandatory_utilization),
    )


@lru_cache(maxsize=None)
def _vectors(job_count: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product(SERVICE_LEVELS, repeat=int(job_count))), dtype=float
    )


def _risk_matrix(
    jobs: pd.DataFrame,
    provider: RiskProvider,
    *,
    method_id: str,
) -> np.ndarray:
    result = np.empty((len(jobs), len(SERVICE_LEVELS)), dtype=float)
    for job_index, row in enumerate(jobs.itertuples(index=False)):
        for level_index, level in enumerate(SERVICE_LEVELS):
            result[job_index, level_index] = provider.risk(
                method_id,
                sample_id=str(row.sample_id),
                task_type=str(row.task_type),
                scene_variant="observed",
                goal_id=str(row.current_goal),
                service_level=float(level),
                delay_ms=float(row.deadline_ms),
            )
    return result


def solve_event(
    event_jobs: pd.DataFrame,
    *,
    provider: RiskProvider,
    profile: DemandProfile,
    candidate: SchedulerCandidate,
    config: Mapping[str, Any],
    variant: str,
) -> EventSolution:
    started = perf_counter()
    jobs = event_jobs.sort_values("sample_id", kind="stable").reset_index(drop=True)
    horizon = float(jobs["deadline_ms"].min())
    capacity = residual_capacity_ms(config, horizon)
    base_ratio = np.asarray(
        [profile.median_ratio[str(value)] for value in jobs["task_type"]],
        dtype=float,
    )
    if variant == "no_reserve":
        planning_ratio = base_ratio
    else:
        planning_ratio = np.asarray(
            [
                profile.ratio(str(value), candidate.reserve_quantile)
                for value in jobs["task_type"]
            ],
            dtype=float,
        )
    c_lo = jobs["c_lo_ms"].to_numpy(dtype=float)
    base_work = c_lo * base_ratio
    planning_work = c_lo * planning_ratio
    vectors = _vectors(len(jobs))
    predicted_work = vectors @ planning_work
    feasible = predicted_work <= capacity + 1e-9
    coverage = np.sum(vectors >= 0.5 - 1e-12, axis=1)
    k_star = int(np.max(coverage[feasible]))
    same_coverage = feasible & (coverage == k_star)
    service = vectors.sum(axis=1)
    s_star = float(np.max(service[same_coverage]))
    envelope = same_coverage & (
        service >= (1.0 - candidate.epsilon_service) * s_star - 1e-9
    )
    risk = _risk_matrix(jobs, provider, method_id=variant)
    level_indexes = np.searchsorted(SERVICE_LEVELS, vectors)
    objective = np.take_along_axis(
        np.broadcast_to(risk[None, :, :], (len(vectors), *risk.shape)),
        level_indexes[:, :, None],
        axis=2,
    ).squeeze(2).sum(axis=1)
    allowed = np.flatnonzero(envelope)
    order = np.lexsort(
        (
            np.asarray(
                ["|".join(f"{value:.2f}" for value in row) for row in vectors[allowed]]
            ),
            -service[allowed],
            objective[allowed],
        )
    )
    selected_index = int(allowed[int(order[0])])
    selected = vectors[selected_index]
    reserve = float(selected @ np.maximum(planning_work - base_work, 0.0))
    return EventSolution(
        event_id=str(jobs.iloc[0]["decision_event_id"]),
        selected_levels=dict(zip(jobs["sample_id"].astype(str), selected, strict=True)),
        k_star=k_star,
        s_star=s_star,
        selected_service=float(selected.sum()),
        physical_capacity_ms=capacity,
        predicted_target_work_ms=float(predicted_work[selected_index]),
        reserve_ms=reserve,
        same_coverage_vectors=int(same_coverage.sum()),
        envelope_vectors=int(envelope.sum()),
        objective_value=float(objective[selected_index]),
        decision_time_ms=(perf_counter() - started) * 1000.0,
    )


def optimize_events(
    visible: pd.DataFrame,
    *,
    provider: RiskProvider,
    profile: DemandProfile,
    candidate: SchedulerCandidate,
    config: Mapping[str, Any],
    variant: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    optional = visible.loc[
        visible["task_class"].isin(("soft_rt", "best_effort"))
    ].copy()
    levels: dict[str, float] = {}
    diagnostics: list[dict[str, Any]] = []
    for event_id, frame in optional.groupby("decision_event_id", sort=True):
        solution = solve_event(
            frame,
            provider=provider,
            profile=profile,
            candidate=candidate,
            config=config,
            variant=variant,
        )
        levels.update(solution.selected_levels)
        diagnostics.append(
            {
                "decision_event_id": str(event_id),
                "variant": variant,
                **{
                    name: value
                    for name, value in solution.__dict__.items()
                    if name not in {"event_id", "selected_levels"}
                },
                "target_vector": "|".join(
                    f"{sample_id}:{level:.2f}"
                    for sample_id, level in sorted(solution.selected_levels.items())
                ),
            }
        )
    return levels, pd.DataFrame(diagnostics)
