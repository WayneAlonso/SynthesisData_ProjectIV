from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ProfileResult:
    summary: dict
    missing_report: pd.DataFrame
    target_distribution: dict


def profile_dataset(df: pd.DataFrame, target_column: str, sensitive_attributes: list[str]) -> ProfileResult:
    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_ratio=lambda x: x["missing_count"] / len(df))
        .reset_index(names="column")
        .sort_values(["missing_ratio", "missing_count"], ascending=False)
    )

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "sensitive_attributes": sensitive_attributes,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    target_distribution = df[target_column].value_counts(dropna=False, normalize=True).round(4).to_dict()
    return ProfileResult(summary=summary, missing_report=missing, target_distribution=target_distribution)
