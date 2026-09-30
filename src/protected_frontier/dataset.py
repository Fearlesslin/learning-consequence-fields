from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import data_root


@dataclass(frozen=True)
class RootFiles:
    visible: Path
    hidden: Path
    payload: Path
    surfaces: Path


def root_files(config: Mapping[str, Any], root_seed: int) -> RootFiles:
    root = data_root(config) / str(int(root_seed))
    return RootFiles(
        visible=root / "jobs_visible.parquet",
        hidden=root / "jobs_hidden.parquet",
        payload=root / "task_payload_manifest.parquet",
        surfaces=root / "consequence_surfaces.npz",
    )


def load_root(
    config: Mapping[str, Any], root_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    paths = root_files(config, root_seed)
    missing = [path for path in paths.__dict__.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    arrays = np.load(paths.surfaces, allow_pickle=False)
    return (
        pd.read_parquet(paths.visible),
        pd.read_parquet(paths.hidden),
        pd.read_parquet(paths.payload),
        arrays["sample_ids"].astype(str),
        arrays["targets"].astype(float),
    )

