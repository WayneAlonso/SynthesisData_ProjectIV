from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import DataLoader
from ..experiments.pipeline import evaluate_dataset
from ..paths import FIGURES_DIR, REPORTS_DIR
from ..settings import Settings
from ..utils.io import timestamp, write_dataframe, write_json, write_markdown
from ..utils.reporting import dataframe_markdown, save_line_chart


def _rebalance_sensitive_groups(df: pd.DataFrame, sensitive_col: str, minority_ratio: float, random_state: int) -> pd.DataFrame:
    majority_value = df[sensitive_col].mode().iloc[0]
    minority_df = df[df[sensitive_col] != majority_value]
    majority_df = df[df[sensitive_col] == majority_value]
    target_minority_size = int(len(df) * minority_ratio)
    resampled_minority = minority_df.sample(
        n=max(1, min(target_minority_size, len(minority_df))),
        replace=len(minority_df) < target_minority_size,
        random_state=random_state,
    )
    remaining_majority = len(df) - len(resampled_minority)
    resampled_majority = majority_df.sample(
        n=max(1, min(remaining_majority, len(majority_df))),
        replace=len(majority_df) < remaining_majority,
        random_state=random_state,
    )
    return pd.concat([resampled_majority, resampled_minority], ignore_index=True).sample(frac=1.0, random_state=random_state)


def _apply_privacy_proxy(df: pd.DataFrame, strength: float, random_state: int) -> pd.DataFrame:
    noisy = df.copy()
    rng = pd.Series(range(len(noisy))).sample(frac=1.0, random_state=random_state).index
    numeric_cols = noisy.select_dtypes(include="number").columns.tolist()
    generator = np.random.default_rng(random_state)
    for col in numeric_cols:
        if noisy[col].nunique() > 2:
            std = float(noisy[col].std() or 0.0)
            noise = pd.Series(index=noisy.index, data=0.0)
            noise.loc[rng] = pd.Series(generator.normal(0, std * strength, size=len(noisy)))
            noisy[col] = noisy[col] + noise
    return noisy


def run_sensitivity(dataset_name: str, source: str = "auto") -> dict:
    settings = Settings()
    dataset_cfg = settings.dataset(dataset_name)
    loader = DataLoader(random_state=settings.random_state)
    loaded = loader.load(dataset_cfg, source=source)
    run_id = f"{dataset_name}_sensitivity_{timestamp()}"

    baseline_params = settings.baseline_model
    results = {"minority_ratio": [], "sample_size": [], "privacy_proxy": []}

    for ratio in settings.sensitivity.get("minority_ratio", []):
        adjusted = _rebalance_sensitive_groups(
            loaded.data,
            sensitive_col=dataset_cfg.sensitive_attributes[0],
            minority_ratio=float(ratio),
            random_state=settings.random_state,
        )
        split_index = max(1, int(len(adjusted) * (1 - settings.test_size)))
        shuffled = adjusted.sample(frac=1.0, random_state=settings.random_state).reset_index(drop=True)
        outcome = evaluate_dataset(
            shuffled.iloc[:split_index].reset_index(drop=True),
            shuffled.iloc[split_index:].reset_index(drop=True),
            dataset_cfg,
            baseline_params,
            settings.random_state,
        )
        results["minority_ratio"].append(
            {
                "minority_ratio": ratio,
                "accuracy": outcome["performance"]["accuracy"],
                "f1": outcome["performance"]["f1"],
            }
        )

    for sample_size in settings.sensitivity.get("sample_size", []):
        subset = loaded.data.sample(
            n=min(int(sample_size), len(loaded.data)),
            replace=len(loaded.data) < int(sample_size),
            random_state=settings.random_state,
        )
        split_index = max(1, int(len(subset) * (1 - settings.test_size)))
        shuffled = subset.sample(frac=1.0, random_state=settings.random_state).reset_index(drop=True)
        outcome = evaluate_dataset(
            shuffled.iloc[:split_index].reset_index(drop=True),
            shuffled.iloc[split_index:].reset_index(drop=True),
            dataset_cfg,
            baseline_params,
            settings.random_state,
        )
        results["sample_size"].append(
            {
                "sample_size": sample_size,
                "accuracy": outcome["performance"]["accuracy"],
                "f1": outcome["performance"]["f1"],
            }
        )

    for strength in settings.sensitivity.get("privacy_proxy", []):
        noisy = _apply_privacy_proxy(loaded.data, strength=float(strength), random_state=settings.random_state)
        split_index = max(1, int(len(noisy) * (1 - settings.test_size)))
        shuffled = noisy.sample(frac=1.0, random_state=settings.random_state).reset_index(drop=True)
        outcome = evaluate_dataset(
            shuffled.iloc[:split_index].reset_index(drop=True),
            shuffled.iloc[split_index:].reset_index(drop=True),
            dataset_cfg,
            baseline_params,
            settings.random_state,
        )
        results["privacy_proxy"].append(
            {
                "privacy_proxy": strength,
                "accuracy": outcome["performance"]["accuracy"],
                "f1": outcome["performance"]["f1"],
            }
        )

    for experiment_name, rows in results.items():
        if rows:
            df = pd.DataFrame(rows)
            x_col = [c for c in df.columns if c not in {"accuracy", "f1"}][0]
            write_dataframe(df, REPORTS_DIR / f"{run_id}_{experiment_name}.csv")
            save_line_chart(
                df,
                x=x_col,
                y_columns=["accuracy", "f1"],
                title=f"{dataset_name} {experiment_name} sensitivity",
                path=FIGURES_DIR / f"{run_id}_{experiment_name}.png",
            )

    markdown = "\n".join(
        [
            f"# Sensitivity Summary: {dataset_name}",
            "",
            f"- run id: `{run_id}`",
            "",
            "## Minority Ratio",
            "",
            dataframe_markdown(pd.DataFrame(results["minority_ratio"]).round(4).fillna("")),
            "",
            "## Sample Size",
            "",
            dataframe_markdown(pd.DataFrame(results["sample_size"]).round(4).fillna("")),
            "",
            "## Privacy Proxy",
            "",
            dataframe_markdown(pd.DataFrame(results["privacy_proxy"]).round(4).fillna("")),
        ]
    )

    write_json(results, REPORTS_DIR / f"{run_id}.json")
    write_markdown(markdown, REPORTS_DIR / f"{run_id}.md")
    return {"run_id": run_id, "dataset": dataset_name, "results": results}
