from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .config import model_root
from .dataset import load_root
from .risk import SurfaceRiskProvider
from .consequence_field import (
    MonotoneConsequenceField,
    ScalarPriorityModel,
    build_feature_examples,
    goal_weights,
    interpolate_components,
)


@dataclass(frozen=True)
class TruthSurfaceProvider:
    surfaces: dict[tuple[str, str], np.ndarray]
    config: Mapping[str, Any]

    def components(
        self,
        sample_id: str,
        *,
        scene_variant: str,
        service_level: float,
        delay_ms: float,
    ) -> np.ndarray:
        return interpolate_components(
            self.surfaces[(str(sample_id), str(scene_variant))],
            service_level=float(service_level),
            delay_ms=float(delay_ms),
        )

    def risk(
        self,
        method_id: str,
        *,
        sample_id: str,
        task_type: str,
        scene_variant: str,
        goal_id: str,
        service_level: float,
        delay_ms: float,
    ) -> float:
        del method_id, task_type
        return float(
            goal_weights(self.config, str(goal_id))
            @ self.components(
                sample_id,
                scene_variant=scene_variant,
                service_level=service_level,
                delay_ms=delay_ms,
            )
        )


def root_examples(
    config: Mapping[str, Any], root_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, TruthSurfaceProvider]:
    visible, _, payload, sample_ids, targets = load_root(config, root_seed)
    examples = build_feature_examples(visible, payload)
    lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    surfaces: dict[tuple[str, str], np.ndarray] = {}
    for row in examples.itertuples(index=False):
        scene_index = 0 if str(row.scene_variant) == "observed" else 1
        surfaces[(str(row.sample_id), str(row.scene_variant))] = targets[
            lookup[str(row.sample_id)], scene_index
        ]
    return examples, visible, TruthSurfaceProvider(surfaces, config)


def load_models(config: Mapping[str, Any]) -> tuple[
    MonotoneConsequenceField, ScalarPriorityModel
]:
    root = model_root(config)
    return (
        joblib.load(root / "consequence_model.joblib"),
        joblib.load(root / "scalar_control.joblib"),
    )


def providers(
    config: Mapping[str, Any],
    model: MonotoneConsequenceField,
    scalar: ScalarPriorityModel,
    examples: pd.DataFrame,
) -> dict[str, Any]:
    full_prediction = model.predict_surface(examples, mode="full")
    static_prediction = model.predict_surface(examples, mode="static")
    keys = list(
        zip(
            examples["sample_id"].astype(str),
            examples["scene_variant"].astype(str),
            strict=True,
        )
    )
    full_surfaces = {key: full_prediction[index] for index, key in enumerate(keys)}
    static_surfaces = {key: static_prediction[index] for index, key in enumerate(keys)}
    mean_goal = np.mean(
        [goal_weights(config, str(item["goal_id"])) for item in config["goals"]],
        axis=0,
    )
    observed = examples.loc[examples["scene_variant"].astype(str).eq("observed")]
    scalar_scores = scalar.predict(observed)

    class ScalarProvider:
        def __init__(self) -> None:
            self.scores = dict(
                zip(observed["sample_id"].astype(str), scalar_scores, strict=True)
            )
            self.deadlines = dict(
                zip(
                    observed["sample_id"].astype(str),
                    observed["deadline_ms"].astype(float),
                    strict=True,
                )
            )

        def risk(
            self,
            method_id: str,
            *,
            sample_id: str,
            service_level: float,
            delay_ms: float,
            **_: Any,
        ) -> float:
            del method_id
            score = max(float(self.scores[str(sample_id)]), 0.0)
            gap = 1.0 - float(np.clip(service_level, 0.0, 1.0))
            delay = float(delay_ms) / max(float(self.deadlines[str(sample_id)]), 1e-9)
            return score * (0.03 + 0.97 * gap) * (1.0 + 0.30 * min(delay, 4.0))

    return {
        "full": SurfaceRiskProvider(surfaces=full_surfaces, config=config),
        "mean_goal": SurfaceRiskProvider(
            surfaces=full_surfaces,
            config=config,
            fixed_goal_weights=mean_goal,
        ),
        "static": SurfaceRiskProvider(surfaces=static_surfaces, config=config),
        "scalar": ScalarProvider(),
    }
