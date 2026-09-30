from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .contracts import DELAYS_MS, LEVELS, LOSS_COMPONENTS, goal_specs


LEVEL_ARRAY = np.asarray(LEVELS, dtype=float)
DELAY_ARRAY = np.asarray(DELAYS_MS, dtype=float)
COMPONENT_COUNT = len(LOSS_COMPONENTS)
GRID_SIZE = len(LEVELS) * len(DELAYS_MS)
COEFFICIENTS_PER_COMPONENT = 24


def _basis() -> np.ndarray:
    matrix = np.zeros((GRID_SIZE, COEFFICIENTS_PER_COMPONENT), dtype=float)
    for deficit_index in range(4):
        for delay_index in range(6):
            row = deficit_index * 6 + delay_index
            matrix[row, 0] = 1.0
            cursor = 1
            for service_step in range(3):
                matrix[row, cursor] = float(service_step < deficit_index)
                cursor += 1
            for delay_step in range(5):
                matrix[row, cursor] = float(delay_step < delay_index)
                cursor += 1
            for service_step in range(3):
                for delay_step in range(5):
                    matrix[row, cursor] = float(
                        service_step < deficit_index and delay_step < delay_index
                    )
                    cursor += 1
    return matrix


MONOTONE_BASIS = _basis()


def _softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def coefficients_to_surface(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)
    deficit = np.einsum("...ck,gk->...cg", coefficients, MONOTONE_BASIS)
    deficit = deficit.reshape(*coefficients.shape[:-2], COMPONENT_COUNT, 4, 6)
    return deficit[..., :, [3, 2, 1, 0], :]


def logits_to_surface(
    logits: np.ndarray,
    *,
    coefficient_scale: float,
    clip: tuple[float, float],
) -> np.ndarray:
    bounded = np.clip(np.asarray(logits, dtype=float), float(clip[0]), float(clip[1]))
    coefficients = float(coefficient_scale) * _softplus(
        bounded.reshape(len(bounded), COMPONENT_COUNT, COEFFICIENTS_PER_COMPONENT)
    )
    return coefficients_to_surface(coefficients)


def scene_vector(row: Mapping[str, Any], variant: str) -> np.ndarray:
    alarm = float(np.clip(float(row.get("causal_alarm_score", 0.0)), 0.0, 1.0))
    control = float(
        np.clip(float(row.get("control_error_proxy", 0.0)) / 4.0, 0.0, 1.0)
    )
    graph = float(
        np.clip(
            max(
                float(row.get("graph_one_hop_activity", 0.0)),
                float(row.get("graph_two_hop_activity", 0.0)),
            ),
            0.0,
            1.0,
        )
    )
    if variant == "observed":
        return np.array([alarm, control, graph], dtype=float)
    return np.array(
        [1.0 - 0.7 * alarm, 0.25 + 0.6 * (1.0 - control), 1.0 - 0.6 * graph],
        dtype=float,
    )


def build_feature_examples(
    visible: pd.DataFrame, payload: pd.DataFrame
) -> pd.DataFrame:
    payload_columns = [
        "sample_id",
        "task_role",
        "payload_pair_id",
        "payload_severity",
        "payload_ambiguity",
        "payload_spatial_extent",
        "payload_temporal_complexity",
        "payload_novelty",
        "payload_evidence_density",
        "defect_type",
    ]
    base = visible.merge(
        payload[payload_columns], on="sample_id", validate="one_to_one"
    )
    base = base.loc[base["task_class"].isin(("soft_rt", "best_effort"))].copy()
    frames: list[pd.DataFrame] = []
    for variant in ("observed", "counterfactual"):
        frame = base.copy()
        vectors = np.vstack(
            [scene_vector(row._asdict(), variant) for row in frame.itertuples(index=False)]
        )
        frame["scene_variant"] = variant
        frame["scene_alarm"] = vectors[:, 0]
        frame["scene_control"] = vectors[:, 1]
        frame["scene_graph"] = vectors[:, 2]
        frame["example_id"] = frame["sample_id"].astype(str) + "::" + variant
        frame["time_block_1s"] = np.floor(
            frame["release_ms"].astype(float) / 1000.0
        ).astype(int)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["sample_id", "scene_variant"], kind="stable"
    ).reset_index(drop=True)


class EmbeddedResidualBlock:
    def __init__(
        self,
        *,
        categorical: Iterable[str],
        numeric: Iterable[str],
        embedding_dim: int,
        hidden_dim: int,
        config: Mapping[str, Any],
        random_state: int,
    ) -> None:
        self.categorical = tuple(categorical)
        self.numeric = tuple(numeric)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.config = config
        self.random_state = int(random_state)
        self.categories: dict[str, tuple[str, ...]] = {}
        self.numeric_mean = np.empty(0)
        self.numeric_scale = np.empty(0)
        self.projector: Any = None
        self.model: Any = None

    def _raw_matrix(self, frame: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            raise RuntimeError("training is not included in this distribution")
        parts: list[np.ndarray] = []
        for column in self.categorical:
            values = frame[column].fillna("<missing>").astype(str)
            levels = self.categories[column]
            lookup = {value: index for index, value in enumerate(levels)}
            one_hot = np.zeros((len(frame), len(levels)), dtype=float)
            indexes = values.map(lookup).fillna(-1).to_numpy(dtype=int)
            valid = indexes >= 0
            one_hot[np.flatnonzero(valid), indexes[valid]] = 1.0
            parts.append(one_hot)
        if self.numeric:
            numeric = (
                frame[list(self.numeric)]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            parts.append((numeric - self.numeric_mean) / self.numeric_scale)
        return np.hstack(parts) if parts else np.ones((len(frame), 1), dtype=float)

    def _embed(self, frame: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        if fit:
            raise RuntimeError("training is not included in this distribution")
        if self.projector is None:
            raise RuntimeError("residual block is not fitted")
        embedded = self.projector.transform(self._raw_matrix(frame))
        if embedded.shape[1] < self.embedding_dim:
            embedded = np.pad(
                embedded,
                ((0, 0), (0, self.embedding_dim - embedded.shape[1])),
            )
        return embedded[:, : self.embedding_dim]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("residual block is not fitted")
        return np.asarray(self.model.predict(self._embed(frame)), dtype=float)


@dataclass
class MonotoneConsequenceField:
    candidate_id: str
    embedding_dim: int
    hidden_dim: int
    config: Mapping[str, Any]
    type_prior: dict[str, np.ndarray]
    global_prior: np.ndarray
    instance_block: EmbeddedResidualBlock
    scene_block: EmbeddedResidualBlock
    interaction_block: EmbeddedResidualBlock
    component_scales: np.ndarray
    stage_a_iterations: dict[str, int]

    def clone(self) -> "MonotoneConsequenceField":
        return copy.deepcopy(self)

    def predict_logits(self, frame: pd.DataFrame, *, mode: str = "full") -> np.ndarray:
        task_types = frame["task_type"].astype(str)
        prior = np.vstack(
            [self.type_prior.get(value, self.global_prior) for value in task_types]
        )
        if mode == "static":
            return prior
        instance = self.instance_block.predict(frame)
        if mode == "instance":
            return prior + instance
        if mode == "scene_mask":
            return prior + instance + self.interaction_block.predict(frame)
        if mode != "full":
            raise ValueError(f"unknown inference mode: {mode}")
        return (
            prior
            + instance
            + self.scene_block.predict(frame)
            + self.interaction_block.predict(frame)
        )

    def predict_surface(
        self,
        frame: pd.DataFrame,
        *,
        mode: str = "full",
        apply_scales: bool = True,
    ) -> np.ndarray:
        surface = logits_to_surface(
            self.predict_logits(frame, mode=mode),
            coefficient_scale=float(self.config["counterfactual"]["coefficient_scale"]),
            clip=tuple(
                float(value) for value in self.config["counterfactual"]["logit_clip"]
            ),
        )
        if apply_scales:
            surface = surface * self.component_scales.reshape(1, -1, 1, 1)
        return surface


def interpolate_components(
    surface: np.ndarray,
    *,
    service_level: float,
    delay_ms: float,
) -> np.ndarray:
    level = float(np.clip(service_level, 0.0, 1.0))
    delay = float(max(delay_ms, 0.0))
    delayed = np.empty((COMPONENT_COUNT, len(LEVELS)), dtype=float)
    for component in range(COMPONENT_COUNT):
        for level_index in range(len(LEVELS)):
            delayed[component, level_index] = np.interp(
                delay, DELAY_ARRAY, surface[component, level_index]
            )
    return np.asarray(
        [
            np.interp(level, LEVEL_ARRAY, delayed[component])
            for component in range(COMPONENT_COUNT)
        ],
        dtype=float,
    )


def component_benefit(
    surface: np.ndarray,
    *,
    old_level: float,
    new_level: float,
    delay_ms: float,
) -> np.ndarray:
    return interpolate_components(
        surface, service_level=old_level, delay_ms=delay_ms
    ) - interpolate_components(surface, service_level=new_level, delay_ms=delay_ms)


def goal_weights(config: Mapping[str, Any], goal_id: str | None) -> np.ndarray:
    if goal_id is None or goal_id == "default":
        return np.asarray(config["default_goal_weights"], dtype=float)
    return np.asarray(goal_specs(config)[str(goal_id)]["weights"], dtype=float)


class ScalarPriorityModel:
    def __init__(self, block: EmbeddedResidualBlock | None = None) -> None:
        self.block = block

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.block is None:
            raise RuntimeError("scalar control is not fitted")
        return self.block.predict(frame).reshape(-1)
