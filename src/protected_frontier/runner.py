from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .baselines import load_method_configs, run_baseline
from .config import model_root, result_root
from .dataset import load_root, root_files
from .metrics import evaluate
from .model import load_models, providers, root_examples
from .optimizer import (
    candidate_from_config,
    load_demand_profile,
    optimize_events,
    solve_event,
)
from .scheduler import run_scheduler


FULL_VARIANTS = {
    "protected_frontier": ("full", "full"),
    "no_reserve": ("no_reserve", "full"),
    "no_goal": ("full", "mean_goal"),
    "scalar_priority": ("full", "scalar"),
    "static_type": ("full", "static"),
}


def scale_workload(
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled_visible = visible.copy()
    scaled_hidden = hidden.copy()
    scaled_visible["c_lo_ms"] = scaled_visible["c_lo_ms"].astype(float) * float(scale)
    for column in ("c_lo_us", "c_hi_us", "exec_budget_us"):
        scaled_visible[column] = (
            scaled_visible[column].astype(float).mul(float(scale)).round().astype(int)
        )
    scaled_hidden["actual_exec_ms"] = (
        scaled_hidden["actual_exec_ms"].astype(float) * float(scale)
    )
    scaled_hidden["actual_exec_us"] = (
        scaled_hidden["actual_exec_ms"].mul(1000.0).round().astype(int)
    )
    return scaled_visible, scaled_hidden


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _run_full_method(
    *,
    method: str,
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    truth: Any,
    provider_set: Mapping[str, Any],
    profile: Any,
    candidate: Any,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variant, provider_name = FULL_VARIANTS[method]
    provider = provider_set[provider_name]
    started = perf_counter()
    levels, events = optimize_events(
        visible,
        provider=provider,
        profile=profile,
        candidate=candidate,
        config=config,
        variant=variant,
    )
    outcomes = run_scheduler(
        visible,
        hidden,
        target_levels=levels,
        provider=provider,
        candidate=candidate,
        config=config,
    )
    metrics, semantic, joined = evaluate(
        visible,
        outcomes,
        truth_provider=truth,
        event_diagnostics=events,
    )
    metrics["run_wall_time_ms"] = (perf_counter() - started) * 1000.0
    return joined, semantic, events, metrics


def _run_baseline_method(
    *,
    method: str,
    visible: pd.DataFrame,
    hidden: pd.DataFrame,
    truth: Any,
    method_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, None, dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    outcomes, artifact = run_baseline(
        method,
        visible,
        hidden,
        method_config,
        channels=int(config["platform"]["channels"]),
    )
    metrics, semantic, joined = evaluate(
        visible,
        outcomes,
        truth_provider=truth,
    )
    metrics["run_wall_time_ms"] = (perf_counter() - started) * 1000.0
    return joined, semantic, None, metrics, artifact


def _save_run(
    output: Path,
    *,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    outcomes: pd.DataFrame,
    semantic: pd.DataFrame,
    events: pd.DataFrame | None,
    save_details: bool,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "run.json", metadata)
    if save_details:
        outcomes.to_parquet(output / "outcomes.parquet", index=False)
        semantic.to_parquet(output / "semantic_outcomes.parquet", index=False)
        if events is not None:
            events.to_parquet(output / "decision_events.parquet", index=False)


def _matrix(config: Mapping[str, Any]) -> list[tuple[float, str]]:
    rows = [
        (float(load), str(method))
        for load in config["evaluation"]["loads"]
        for method in config["evaluation"]["methods"]
    ]
    rows.extend(
        (1.0, str(method))
        for method in (
            list(config["evaluation"]["ablations"])
            + list(config["evaluation"]["controls"])
        )
    )
    return rows


def run_experiments(
    config: Mapping[str, Any],
    *,
    roots: Iterable[int] | None = None,
    loads: Iterable[float] | None = None,
    methods: Iterable[str] | None = None,
    save_details: bool = False,
) -> pd.DataFrame:
    selected_roots = set(int(value) for value in roots) if roots else None
    selected_loads = set(float(value) for value in loads) if loads else None
    selected_methods = set(str(value) for value in methods) if methods else None
    matrix = [
        (load, method)
        for load, method in _matrix(config)
        if (selected_loads is None or load in selected_loads)
        and (selected_methods is None or method in selected_methods)
    ]
    model, scalar = load_models(config)
    profile = load_demand_profile(model_root(config) / "demand_profile.json")
    candidate = candidate_from_config(config)
    baseline_configs = load_method_configs(config)
    output_root = result_root(config) / "runs"
    records: list[dict[str, Any]] = []
    for root_value in config["roots"]:
        root_seed = int(root_value)
        if selected_roots is not None and root_seed not in selected_roots:
            continue
        examples, visible, truth = root_examples(config, root_seed)
        _, hidden, _, _, _ = load_root(config, root_seed)
        provider_set = providers(config, model, scalar, examples)
        scaled: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for load, method in matrix:
            if load not in scaled:
                scaled[load] = scale_workload(visible, hidden, load)
            run_visible, run_hidden = scaled[load]
            output = (
                output_root
                / str(root_seed)
                / f"load_{load:.2f}"
                / method
            )
            artifact: dict[str, Any] = {}
            if method in FULL_VARIANTS:
                outcomes, semantic, events, metrics = _run_full_method(
                    method=method,
                    visible=run_visible,
                    hidden=run_hidden,
                    truth=truth,
                    provider_set=provider_set,
                    profile=profile,
                    candidate=candidate,
                    config=config,
                )
            else:
                method_config = baseline_configs.get(method, {})
                outcomes, semantic, events, metrics, artifact = _run_baseline_method(
                    method=method,
                    visible=run_visible,
                    hidden=run_hidden,
                    truth=truth,
                    method_config=method_config,
                    config=config,
                )
            metadata = {
                "root_seed": root_seed,
                "load": load,
                "method": method,
                "candidate": candidate.candidate_id if method in FULL_VARIANTS else "fixed",
                "artifact": artifact,
            }
            _save_run(
                output,
                metrics=metrics,
                metadata=metadata,
                outcomes=outcomes,
                semantic=semantic,
                events=events,
                save_details=save_details,
            )
            records.append(
                {
                    "root_seed": root_seed,
                    "load": load,
                    "method": method,
                    **metrics,
                }
            )
            print(
                f"root={root_seed} load={load:.2f} method={method}",
                flush=True,
            )
    frame = pd.DataFrame(records)
    aggregate_path = result_root(config) / "metrics_by_run.csv"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    if aggregate_path.is_file():
        previous = pd.read_csv(aggregate_path)
        frame = pd.concat([previous, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["root_seed", "load", "method"], keep="last"
        )
    frame = frame.sort_values(["root_seed", "load", "method"], kind="stable")
    frame.to_csv(aggregate_path, index=False)
    return frame.reset_index(drop=True)


def check_package(config: Mapping[str, Any]) -> dict[str, int]:
    model, scalar = load_models(config)
    candidate = candidate_from_config(config)
    profile = load_demand_profile(model_root(config) / "demand_profile.json")
    jobs = 0
    for root_value in config["roots"]:
        paths = root_files(config, int(root_value))
        if not all(path.is_file() for path in paths.__dict__.values()):
            raise FileNotFoundError(f"incomplete root {root_value}")
        visible, hidden, payload, sample_ids, surfaces = load_root(
            config, int(root_value)
        )
        if not (
            len(visible) == len(hidden) == len(payload)
            and len(sample_ids) == len(surfaces)
            and set(sample_ids).issubset(set(visible["sample_id"].astype(str)))
        ):
            raise ValueError(f"row mismatch in root {root_value}")
        jobs += len(visible)
    first_root = int(config["roots"][0])
    examples, visible, _ = root_examples(config, first_root)
    provider = providers(config, model, scalar, examples)["full"]
    event = next(
        frame
        for _, frame in visible.loc[
            visible["task_class"].isin(("soft_rt", "best_effort"))
        ].groupby("decision_event_id", sort=True)
    )
    solve_event(
        event,
        provider=provider,
        profile=profile,
        candidate=candidate,
        config=config,
        variant="full",
    )
    return {"roots": len(config["roots"]), "jobs": jobs, "models": 2}
