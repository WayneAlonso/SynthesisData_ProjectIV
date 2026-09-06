from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR


@dataclass
class DatasetConfig:
    name: str
    path: str
    target_column: str
    sensitive_attributes: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]
    target_transform: dict[str, Any] | None = None
    excluded_feature_columns: list[str] | None = None
    na_values: list[str] | None = None
    role: str = "train"


class Settings:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or CONFIGS_DIR / "experiment.json"
        with self.config_path.open("r", encoding="utf-8") as fh:
            self.raw: dict[str, Any] = json.load(fh)

    @property
    def random_state(self) -> int:
        return int(self.raw.get("random_state", 42))

    @property
    def test_size(self) -> float:
        return float(self.raw.get("test_size", 0.2))

    @property
    def baseline_model(self) -> dict[str, Any]:
        return dict(self.raw.get("baseline_model", {}))

    @property
    def synthesis_models_config(self) -> dict[str, Any]:
        return dict(self.raw.get("synthesis_models", {}))

    @property
    def comparison_models(self) -> dict[str, Any]:
        return dict(self.raw.get("comparison_models", {}))

    @property
    def sensitive_attributes(self) -> list[str]:
        return list(self.raw.get("sensitive_attributes", []))

    @property
    def n_repeats(self) -> int:
        return int(self.raw.get("n_repeats", 1))

    @property
    def optuna_n_trials(self) -> int:
        return int(self.raw.get("optuna", {}).get("n_trials", 50))

    @property
    def optuna_search_space(self) -> dict:
        return dict(self.raw.get("optuna", {}).get("search_space", {}))

    @property
    def fairlearn_lambdas(self) -> list[float]:
        return list(self.raw.get("fairlearn", {}).get("lambdas", [0.1, 0.5, 1.0]))

    @property
    def fairlearn_config(self) -> dict:
        return dict(self.raw.get("fairlearn", {}))

    @property
    def n_synth(self) -> list[str]:
        return list(self.raw.get("synthesis_models", {}).keys())

    @property
    def privacy_levels(self) -> list[float]:
        return list(self.raw.get("privacy_levels", [0.0, 0.1, 0.3]))

    @property
    def sensitivity(self) -> dict[str, list[float]]:
        return dict(self.raw.get("sensitivity", {}))

    def dataset(self, name: str) -> DatasetConfig:
        dataset_cfg = dict(self.raw["dataset_catalog"][name])
        dataset_cfg.setdefault("role", "train")
        return DatasetConfig(name=name, **dataset_cfg)
