from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import os


def _build_metadata(df: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    return metadata


@dataclass
class SynthesisResult:
    model_name: str
    synthetic_data: pd.DataFrame
    backend: str


class BootstrapSynthesizer:
    def __init__(self, random_state: int = 42, noise_scale: float = 0.03) -> None:
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.train_df: pd.DataFrame | None = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df.copy().reset_index(drop=True)

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.train_df is None:
            raise RuntimeError("Synthesizer must be fit before sampling.")
        sampled = self.train_df.sample(n=num_rows, replace=True, random_state=self.random_state).reset_index(drop=True)
        numeric_cols = sampled.select_dtypes(include=["number"]).columns
        for column in numeric_cols:
            if sampled[column].nunique() > 2:
                std = float(sampled[column].std() or 0.0)
                sampled[column] = sampled[column] + self.rng.normal(0, std * self.noise_scale, size=len(sampled))
        return sampled


class SDVTableSynthesizer:
    def __init__(self, model_name: str, params: dict | None = None, random_state: int = 42,
                 allow_fallback: bool = False) -> None:
        self.model_name = model_name
        self.params = params or {}
        self.random_state = random_state
        self.allow_fallback = allow_fallback
        self.model = None
        self.backend = "uninitialized"
        self.fallback = BootstrapSynthesizer(random_state=random_state)

    def fit(self, train_df: pd.DataFrame) -> None:
        try:
            params = dict(self.params)
            if os.environ.get("W4_USE_GPU", "0") == "1" and self.model_name in {"ctgan", "tvae", "copula_gan"}:
                params.setdefault("enable_gpu", True)
            if self.model_name == "ctgan":
                from sdv.single_table import CTGANSynthesizer

                self.model = CTGANSynthesizer(metadata=_build_metadata(train_df), **params)
            elif self.model_name == "tvae":
                from sdv.single_table import TVAESynthesizer

                self.model = TVAESynthesizer(metadata=_build_metadata(train_df), **params)
            elif self.model_name == "gaussian_copula":
                from sdv.single_table import GaussianCopulaSynthesizer

                self.model = GaussianCopulaSynthesizer(metadata=_build_metadata(train_df), **params)
            elif self.model_name == "copula_gan":
                from sdv.single_table import CopulaGANSynthesizer

                self.model = CopulaGANSynthesizer(metadata=_build_metadata(train_df), **params)
            else:
                raise ValueError(f"Unknown synthesizer: {self.model_name}")

            self.model.fit(train_df)
            self.backend = "sdv"
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"SDV synthesizer '{self.model_name}' failed; refusing silent fallback: {exc}"
                ) from exc
            self.fallback.fit(train_df)
            self.backend = "bootstrap_fallback"

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.backend == "sdv" and self.model is not None:
            return self.model.sample(num_rows=num_rows)
        return self.fallback.sample(num_rows)

    def run(self, train_df: pd.DataFrame, num_rows: int) -> SynthesisResult:
        self.fit(train_df)
        synthetic_data = self.sample(num_rows=num_rows)
        return SynthesisResult(
            model_name=self.model_name,
            synthetic_data=synthetic_data,
            backend=self.backend,
        )
