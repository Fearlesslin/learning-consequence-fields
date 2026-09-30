from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .consequence_field import DELAY_ARRAY, LEVEL_ARRAY, goal_weights, interpolate_components


class SurfaceRiskProvider:
    def __init__(
        self,
        *,
        surfaces: Mapping[tuple[str, str], np.ndarray],
        config: Mapping[str, Any],
        fixed_goal_weights: np.ndarray | None = None,
    ) -> None:
        self.surfaces = {
            (str(key[0]), str(key[1])): np.asarray(value, dtype=float)
            for key, value in surfaces.items()
        }
        self.config = config
        self.fixed_goal_weights = (
            None
            if fixed_goal_weights is None
            else np.asarray(fixed_goal_weights, dtype=float).copy()
        )
        if self.fixed_goal_weights is not None and self.fixed_goal_weights.shape != (6,):
            raise ValueError("fixed goal vector must contain six weights")
        self._projected: dict[tuple[str, str, str], np.ndarray] = {}
        self._cache: dict[tuple[str, str, str, float, float], float] = {}

    def _weights(self, goal_id: str) -> np.ndarray:
        if self.fixed_goal_weights is not None:
            return self.fixed_goal_weights
        return goal_weights(self.config, goal_id)

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
        goal_key = "fixed" if self.fixed_goal_weights is not None else str(goal_id)
        key = (
            str(sample_id),
            str(scene_variant),
            goal_key,
            float(service_level),
            float(delay_ms),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        projection_key = (str(sample_id), str(scene_variant), goal_key)
        projected = self._projected.get(projection_key)
        if projected is None:
            projected = np.einsum(
                "c,cqd->qd",
                self._weights(str(goal_id)),
                self.surfaces[(str(sample_id), str(scene_variant))],
            )
            self._projected[projection_key] = projected
        delayed = np.asarray(
            [np.interp(float(delay_ms), DELAY_ARRAY, row) for row in projected],
            dtype=float,
        )
        value = float(
            np.interp(
                float(np.clip(service_level, 0.0, 1.0)),
                LEVEL_ARRAY,
                delayed,
            )
        )
        self._cache[key] = value
        return value
