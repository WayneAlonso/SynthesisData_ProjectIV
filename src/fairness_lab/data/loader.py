from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import RAW_DATA_DIR
from ..settings import DatasetConfig


@dataclass
class LoadedDataset:
    name: str
    data: pd.DataFrame
    source: str


def _create_demo_adult_data(rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
    """Mimic the UCI Adult dataset for demo runs."""
    age = rng.integers(18, 70, size=n_samples)
    sex = rng.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])
    race = rng.choice(
        ["White", "Black", "Asian", "Hispanic", "Other"],
        size=n_samples,
        p=[0.62, 0.14, 0.08, 0.11, 0.05],
    )
    education = rng.choice(
        ["HS", "Some College", "Bachelor", "Master", "Doctorate"],
        size=n_samples,
        p=[0.28, 0.27, 0.24, 0.15, 0.06],
    )
    workclass = rng.choice(["Private", "Self-emp", "Government", "Other"], size=n_samples)
    occupation = rng.choice(["Tech", "Sales", "Admin", "Service", "Labor"], size=n_samples)
    hours_per_week = rng.integers(20, 60, size=n_samples)
    capital_gain = rng.gamma(1.5, 2500, size=n_samples).round(0)
    capital_loss = rng.gamma(1.2, 300, size=n_samples).round(0)

    base_score = (
        0.025 * age
        + 0.04 * (hours_per_week - 35)
        + 0.0001 * capital_gain
        - 0.0003 * capital_loss
        + pd.Series(education).map(
            {"HS": -0.3, "Some College": -0.1, "Bachelor": 0.2, "Master": 0.45, "Doctorate": 0.6}
        ).to_numpy()
        + pd.Series(race).map(
            {"White": 0.15, "Black": -0.18, "Asian": 0.1, "Hispanic": -0.12, "Other": -0.05}
        ).to_numpy()
        + pd.Series(sex).map({"Male": 0.08, "Female": 0.0}).to_numpy()
    )

    prob = 1 / (1 + np.exp(-(base_score - 1.8)))
    income = rng.binomial(1, np.clip(prob, 0.01, 0.99))

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "race": race,
            "education": education,
            "workclass": workclass,
            "occupation": occupation,
            "hours_per_week": hours_per_week,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "income_above_50k": income,
        }
    )
    mask = rng.choice([True, False], size=n_samples, p=[0.03, 0.97])
    df.loc[mask, "occupation"] = np.nan
    return df


def _create_demo_sbo_data(rng: np.random.Generator, n_samples: int, sex: np.ndarray, race: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "years_in_business": rng.integers(1, 35, size=n_samples),
            "employees": rng.integers(1, 180, size=n_samples),
            "revenue": rng.normal(280000, 120000, size=n_samples).clip(10000, None).round(0),
            "sex": sex,
            "race": race,
            "industry": rng.choice(["Retail", "Tech", "Health", "Food"], size=n_samples),
            "region": rng.choice(["East", "West", "North", "South"], size=n_samples),
            "loan_approved": rng.binomial(1, 0.4, size=n_samples),
        }
    )


def _create_demo_acs_employment_data(rng: np.random.Generator, n_samples: int, base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    df["employed"] = (
        (df["hours_per_week"] > 30).astype(int)
        | df["education"].isin(["Bachelor", "Master", "Doctorate"])
    ).astype(int)
    return df.drop(columns=["income_above_50k", "capital_gain", "capital_loss"])


def _create_demo_acs_data(rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
    """Mimic the NIST ACS public-use microdata for demo runs."""
    age = rng.integers(18, 90, size=n_samples)
    work_hours = rng.integers(1, 70, size=n_samples)
    sex = rng.choice([1, 2], size=n_samples, p=[0.51, 0.49])
    race = rng.choice([1, 2, 3, 6, 8, 9], size=n_samples, p=[0.60, 0.13, 0.06, 0.10, 0.05, 0.06])
    hisp = rng.choice([1, 2, 24], size=n_samples, p=[0.78, 0.17, 0.05])
    education = rng.choice([16, 18, 19, 20, 21, 22, 23, 24], size=n_samples)
    cow = rng.choice([1, 2, 3, 4, 5, 6, 7], size=n_samples)
    mar = rng.choice([1, 2, 3, 4, 5], size=n_samples)
    occp = rng.choice([425, 1021, 2015, 3601, 4720, 9130], size=n_samples)
    pobp = rng.choice([1, 6, 12, 36, 48, 72, 100, 200], size=n_samples)
    relp = rng.choice([0, 1, 2, 3, 4, 16, 17], size=n_samples)

    income_signal = (
        350 * age
        + 500 * work_hours
        + pd.Series(education).map({16: 0, 18: 3000, 19: 7000, 20: 12000, 21: 18000, 22: 24000, 23: 32000, 24: 42000}).to_numpy()
        + pd.Series(race).map({1: 6000, 2: -5000, 3: 3500, 6: -2500, 8: -1500, 9: -1000}).to_numpy()
        + pd.Series(sex).map({1: 2500, 2: 0}).to_numpy()
        + rng.normal(0, 12000, size=n_samples)
    )
    pincp = np.clip(income_signal, 0, None).round(0)

    df = pd.DataFrame(
        {
            "AGEP": age,
            "WKHP": work_hours,
            "COW": cow,
            "SCHL": education,
            "MAR": mar,
            "OCCP": occp,
            "POBP": pobp,
            "RELP": relp,
            "SEX": sex,
            "RAC1P": race,
            "HISP": hisp,
            "PINCP": pincp,
        }
    )
    df["HIGH_INCOME"] = (df["PINCP"] > 50000).astype(int)
    mask = rng.choice([True, False], size=n_samples, p=[0.02, 0.98])
    df.loc[mask, "OCCP"] = np.nan
    return df


class DataLoader:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def load(self, dataset_cfg: DatasetConfig, source: str = "auto") -> LoadedDataset:
        if source in {"auto", "file"}:
            file_path = Path(dataset_cfg.path)
            if not file_path.is_absolute():
                file_path = RAW_DATA_DIR.parent.parent / dataset_cfg.path
            if file_path.exists():
                df = pd.read_csv(file_path, na_values=dataset_cfg.na_values, low_memory=False)
                df = self._apply_target_transform(df, dataset_cfg)
                return LoadedDataset(
                    name=dataset_cfg.name,
                    data=df,
                    source="file",
                )
        df = create_demo_data(dataset_cfg.name, n_samples=5000, random_state=self.random_state)
        df = self._align_columns(df, dataset_cfg)
        return LoadedDataset(
            name=dataset_cfg.name,
            data=df,
            source="demo",
        )

    def _align_columns(self, df: pd.DataFrame, dataset_cfg: DatasetConfig) -> pd.DataFrame:
        """Map the demo dataframe's columns onto the schema declared in `experiment.json`.

        The demo builder produces a dataset using a small set of English columns; the
        experiment config, however, uses the canonical column names (e.g. EMPLOYMENT_NOISY,
        FIPST, RACE1 for SBO; AGEP, SEX, RAC1P, HISP for ACS). When the demo is loaded we
        align the demo columns onto the configured schema so downstream preprocessing /
        modelling can find them.
        """
        out = df.copy()
        schema = dataset_cfg.categorical_columns + dataset_cfg.numerical_columns + [
            dataset_cfg.target_column
        ]
        demo_to_schema = _DEMO_COLUMN_ALIASES.get(dataset_cfg.name, {})
        for demo_col, schema_col in demo_to_schema.items():
            if demo_col in out.columns and schema_col not in out.columns:
                out[schema_col] = out[demo_col]
        present = [c for c in schema if c in out.columns]
        if present:
            missing = [c for c in schema if c not in out.columns]
            if missing and len(present) >= 4:
                pass
        return out

    def _apply_target_transform(self, df: pd.DataFrame, dataset_cfg: DatasetConfig) -> pd.DataFrame:
        transformed = df.copy()
        rule = dataset_cfg.target_transform or {}
        if dataset_cfg.target_column in transformed.columns:
            return transformed
        if rule.get("type") == "threshold":
            source_column = rule["source_column"]
            threshold = float(rule["threshold"])
            positive_label = int(rule.get("positive_label", 1))
            negative_label = int(rule.get("negative_label", 0))
            if source_column not in transformed.columns:
                raise KeyError(
                    f"Target source column '{source_column}' was not found in dataset '{dataset_cfg.name}'."
                )
            transformed[dataset_cfg.target_column] = np.where(
                pd.to_numeric(transformed[source_column], errors="coerce") > threshold,
                positive_label,
                negative_label,
            )
        return transformed


def create_demo_data(dataset_name: str, n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Top-level helper for generating demo data for a given dataset name."""
    rng = np.random.default_rng(random_state)
    if dataset_name.startswith("acs_") and dataset_name.endswith("_2019"):
        return _create_demo_acs_data(rng, n_samples)
    adult = _create_demo_adult_data(rng, n_samples)
    if dataset_name == "sbo":
        return _create_demo_sbo_data(rng, n_samples, adult["sex"].to_numpy(), adult["race"].to_numpy())
    if dataset_name == "acs_employment":
        return _create_demo_acs_employment_data(rng, n_samples, adult)
    return adult


# Maps demo output columns to the canonical column names declared in experiment.json.
# When the canonical name is not present in the demo dataframe it is created via this mapping.
_DEMO_COLUMN_ALIASES: dict[str, dict[str, str]] = {
    "sbo": {
        "years_in_business": "EMPLOYMENT_NOISY",
        "employees": "PAYROLL_NOISY",
        "revenue": "RECEIPTS_NOISY",
        "race": "RACE1",
        "sex": "SEX1",
        "industry": "SECTOR",
        "region": "ETH1",
    },
}