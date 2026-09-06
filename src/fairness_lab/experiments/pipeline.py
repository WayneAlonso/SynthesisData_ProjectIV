from __future__ import annotations

import pandas as pd

from ..data.loader import DataLoader
from ..data.preprocessing import DataPreprocessor
from ..data.profiling import profile_dataset
from ..fairness.metrics import equal_opportunity_difference, statistical_parity_difference
from ..models.baseline import BaselineClassifier
from ..models.generative import SDVTableSynthesizer
from ..paths import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR
from ..settings import Settings
from ..utils.io import ensure_dir, timestamp, write_dataframe, write_json, write_markdown
from ..utils.reporting import (
    dataframe_markdown,
    save_bar_chart,
    save_heatmap,
    save_scatter_chart,
)


def compute_quality_report(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict[str, float]:
    shared_numeric = [col for col in real_df.select_dtypes(include="number").columns if col in synthetic_df.columns]
    shared_categorical = [col for col in real_df.columns if col in synthetic_df.columns and col not in shared_numeric]

    numeric_shift = 0.0
    if shared_numeric:
        diffs = []
        for col in shared_numeric:
            diffs.append(abs(real_df[col].mean() - synthetic_df[col].mean()))
        numeric_shift = float(sum(diffs) / len(diffs))

    category_overlap = 0.0
    if shared_categorical:
        overlaps = []
        for col in shared_categorical:
            real_values = set(real_df[col].dropna().astype(str).unique())
            synth_values = set(synthetic_df[col].dropna().astype(str).unique())
            overlaps.append(len(real_values & synth_values) / max(1, len(real_values | synth_values)))
        category_overlap = float(sum(overlaps) / len(overlaps))

    return {
        "numeric_mean_shift": numeric_shift,
        "categorical_overlap": category_overlap,
    }


def evaluate_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, dataset_cfg, baseline_params: dict, random_state: int) -> dict:
    prep = DataPreprocessor(dataset_cfg, random_state=random_state)
    prepared_train_test = prep.prepare_from_split(train_df=train_df, test_df=test_df)
    baseline = BaselineClassifier(params=baseline_params, random_state=random_state)
    baseline.fit(prepared_train_test.X_train, prepared_train_test.y_train)
    performance = baseline.evaluate(prepared_train_test.X_test, prepared_train_test.y_test)
    y_pred = pd.Series(baseline.predict(prepared_train_test.X_test))

    fairness = {}
    for attr, sensitive_series in prepared_train_test.sensitive_test.items():
        fairness[attr] = {
            **statistical_parity_difference(y_pred, sensitive_series),
            **equal_opportunity_difference(prepared_train_test.y_test, y_pred, sensitive_series),
        }
    return {"performance": performance, "fairness": fairness}


def _fairness_summary_row(model_name: str, backend: str, fairness: dict[str, dict[str, float]]) -> dict[str, float | str]:
    spd_ranges = [metrics.get("spd_range", 0.0) for metrics in fairness.values()]
    eod_ranges = [metrics.get("eod_tpr_range", 0.0) for metrics in fairness.values()]
    return {
        "model": model_name,
        "backend": backend,
        "spd_range_mean": float(sum(spd_ranges) / max(1, len(spd_ranges))),
        "eod_tpr_range_mean": float(sum(eod_ranges) / max(1, len(eod_ranges))),
    }


def _fairness_detail_rows(model_name: str, backend: str, fairness: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
    rows = []
    for attribute, metrics in fairness.items():
        rows.append(
            {
                "model": model_name,
                "backend": backend,
                "attribute": attribute,
                "spd_range": float(metrics.get("spd_range", 0.0)),
                "eod_tpr_range": float(metrics.get("eod_tpr_range", 0.0)),
            }
        )
    return rows


def run_full_pipeline(dataset_name: str, source: str = "auto") -> dict:
    settings = Settings()
    dataset_cfg = settings.dataset(dataset_name)

    ensure_dir(REPORTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(PROCESSED_DIR)
    run_id = f"{dataset_name}_{timestamp()}"

    loader = DataLoader(random_state=settings.random_state)
    loaded = loader.load(dataset_cfg, source=source)

    profile = profile_dataset(
        loaded.data,
        target_column=dataset_cfg.target_column,
        sensitive_attributes=dataset_cfg.sensitive_attributes,
    )

    preprocessor = DataPreprocessor(
        dataset_cfg=dataset_cfg,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    prepared = preprocessor.prepare(loaded.data)

    write_dataframe(prepared.train_df, PROCESSED_DIR / f"{run_id}_train.csv")
    write_dataframe(prepared.test_df, PROCESSED_DIR / f"{run_id}_test.csv")

    baseline = BaselineClassifier(params=settings.baseline_model, random_state=settings.random_state)
    baseline.fit(prepared.X_train, prepared.y_train)
    baseline_metrics = baseline.evaluate(prepared.X_test, prepared.y_test)
    y_pred = pd.Series(baseline.predict(prepared.X_test))

    baseline_fairness = {}
    for attr, sensitive_series in prepared.sensitive_test.items():
        baseline_fairness[attr] = {
            **statistical_parity_difference(y_pred, sensitive_series),
            **equal_opportunity_difference(prepared.y_test, y_pred, sensitive_series),
        }

    fairness_summary_rows = [_fairness_summary_row("baseline", baseline.backend_name, baseline_fairness)]
    fairness_detail_rows = _fairness_detail_rows("baseline", baseline.backend_name, baseline_fairness)
    experiment_rows = [{**fairness_summary_rows[0], "model": "baseline", "backend": baseline.backend_name, **baseline_metrics}]

    synthesis_reports = {}
    for model_name, model_params in settings.synthesis_models_config.items():
        synthesizer = SDVTableSynthesizer(model_name=model_name, params=model_params, random_state=settings.random_state)
        synthesis_result = synthesizer.run(prepared.train_df, num_rows=len(prepared.train_df))
        synthetic_df = synthesis_result.synthetic_data
        write_dataframe(synthetic_df, PROCESSED_DIR / f"{run_id}_{model_name}_synthetic.csv")

        synthetic_eval = evaluate_dataset(
            train_df=synthetic_df,
            test_df=prepared.test_df,
            dataset_cfg=dataset_cfg,
            baseline_params=settings.baseline_model,
            random_state=settings.random_state,
        )
        quality = compute_quality_report(prepared.train_df, synthetic_df)
        synthesis_reports[model_name] = {
            "backend": synthesis_result.backend,
            "quality": quality,
            **synthetic_eval,
        }
        fairness_summary = _fairness_summary_row(model_name, synthesis_result.backend, synthetic_eval["fairness"])
        fairness_summary_rows.append(fairness_summary)
        fairness_detail_rows.extend(_fairness_detail_rows(model_name, synthesis_result.backend, synthetic_eval["fairness"]))
        experiment_rows.append(
            {
                "model": model_name,
                "backend": synthesis_result.backend,
                **synthetic_eval["performance"],
                "numeric_mean_shift": quality["numeric_mean_shift"],
                "categorical_overlap": quality["categorical_overlap"],
                "spd_range_mean": fairness_summary["spd_range_mean"],
                "eod_tpr_range_mean": fairness_summary["eod_tpr_range_mean"],
            }
        )

    summary_df = pd.DataFrame(experiment_rows)
    fairness_summary_df = pd.DataFrame(fairness_summary_rows)
    fairness_detail_df = pd.DataFrame(fairness_detail_rows)
    missing_df = profile.missing_report.copy()
    target_distribution_df = pd.DataFrame(
        [{"target_value": key, "share": value} for key, value in profile.target_distribution.items()]
    )

    write_dataframe(summary_df, REPORTS_DIR / f"{run_id}_benchmark.csv")
    write_dataframe(fairness_summary_df, REPORTS_DIR / f"{run_id}_fairness_summary.csv")
    write_dataframe(fairness_detail_df, REPORTS_DIR / f"{run_id}_fairness_detail.csv")
    write_dataframe(missing_df, REPORTS_DIR / f"{run_id}_missing_report.csv")
    write_dataframe(target_distribution_df, REPORTS_DIR / f"{run_id}_target_distribution.csv")

    save_bar_chart(summary_df, x="model", y="accuracy", title=f"{dataset_name} accuracy comparison", path=FIGURES_DIR / f"{run_id}_accuracy.png")
    save_bar_chart(summary_df, x="model", y="f1", title=f"{dataset_name} F1 comparison", path=FIGURES_DIR / f"{run_id}_f1.png")
    save_bar_chart(
        summary_df,
        x="model",
        y="categorical_overlap",
        title=f"{dataset_name} categorical overlap comparison",
        path=FIGURES_DIR / f"{run_id}_categorical_overlap.png",
    )
    save_bar_chart(
        summary_df,
        x="model",
        y="numeric_mean_shift",
        title=f"{dataset_name} numeric mean shift comparison",
        path=FIGURES_DIR / f"{run_id}_numeric_shift.png",
    )
    save_scatter_chart(
        summary_df,
        x="spd_range_mean",
        y="accuracy",
        label_col="model",
        title=f"{dataset_name} utility-fairness trade-off",
        path=FIGURES_DIR / f"{run_id}_utility_fairness_tradeoff.png",
    )

    fairness_spd_heatmap = fairness_detail_df.pivot(index="attribute", columns="model", values="spd_range")
    fairness_eod_heatmap = fairness_detail_df.pivot(index="attribute", columns="model", values="eod_tpr_range")
    save_heatmap(
        fairness_spd_heatmap,
        title=f"{dataset_name} SPD range heatmap",
        path=FIGURES_DIR / f"{run_id}_spd_heatmap.png",
    )
    save_heatmap(
        fairness_eod_heatmap,
        title=f"{dataset_name} EOD TPR range heatmap",
        path=FIGURES_DIR / f"{run_id}_eod_heatmap.png",
    )

    result = {
        "run_id": run_id,
        "dataset": dataset_name,
        "source": loaded.source,
        "profile_summary": profile.summary,
        "target_distribution": profile.target_distribution,
        "baseline": {
            "performance": baseline_metrics,
            "fairness": baseline_fairness,
        },
        "synthetic_models": synthesis_reports,
    }

    write_json(result, REPORTS_DIR / f"{run_id}_summary.json")
    markdown = "\n".join(
        [
            f"# Experiment Summary: {dataset_name}",
            "",
            f"- run id: `{run_id}`",
            f"- data source: `{loaded.source}`",
            f"- rows: `{profile.summary['rows']}`",
            f"- columns: `{profile.summary['columns']}`",
            "",
            "## Benchmark Table",
            "",
            dataframe_markdown(summary_df.round(4).fillna("")),
            "",
            "## Fairness Summary Table",
            "",
            dataframe_markdown(fairness_summary_df.round(4).fillna("")),
            "",
            "## Missing Value Report",
            "",
            dataframe_markdown(missing_df.head(20).round(4)),
        ]
    )
    write_markdown(markdown, REPORTS_DIR / f"{run_id}_summary.md")
    return result
