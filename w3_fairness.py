"""W3 - Fairness: fairlearn DP/EO + ThresholdOptimizer + intersectional groups + Pareto."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from fairness_lab.data.loader import DataLoader
from fairness_lab.data.preprocessing import DataPreprocessor
from fairness_lab.fairness.metrics import full_fairness_report
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.settings import Settings
from fairness_lab.utils.io import ensure_dir, write_dataframe, write_json, write_markdown


class W3Fairness:
    """Week 3: fairness-aware training + post-processing + Pareto frontier."""

    def __init__(self, out_dir: Path | None = None, lambdas: list[float] | None = None) -> None:
        self.out_dir = ensure_dir(out_dir or (ARTIFACTS_DIR / "w3"))
        self.lambdas = lambdas or [0.1, 0.5, 1.0]
        self.settings = Settings()

    def run(self, datasets: list[str]) -> None:
        self._require_fairlearn()
        print(f"[W3] datasets={datasets}, lambdas={self.lambdas}")
        rows: list[dict] = []
        for ds_name in datasets:
            print(f"[W3] -> {ds_name}")
            try:
                rows.extend(self._run_one(ds_name))
            except Exception as exc:
                print(f"  !! {ds_name} failed: {exc}")

        if not rows:
            print("[W3] No dataset produced results.")
            return

        df = pd.DataFrame(rows)
        write_dataframe(df, self.out_dir / "fairness_vs_accuracy.csv")
        write_json(self.lambdas, self.out_dir / "lambdas.json")

        pareto = self._select_pareto(df)
        write_dataframe(pareto, self.out_dir / "pareto_frontier.csv")

        report = self._build_markdown(df, pareto)
        write_markdown(report, self.out_dir / "W3_FAIRNESS.md")
        print(f"[W3] wrote artefacts to {self.out_dir}")

    def _load_prepared(self, ds_name: str):
        cfg = self.settings.dataset(ds_name)
        loader = DataLoader(random_state=self.settings.random_state)
        loaded = loader.load(cfg)
        # SBO datasets are 16k x 130 cols 闂?sample down for fairlearn tractability
        if ds_name in {"sbo", "sbo_withheld"} and len(loaded.data) > 15000:
            loaded.data = loaded.data.sample(n=15000, random_state=self.settings.random_state).reset_index(drop=True)
        prep = DataPreprocessor(dataset_cfg=cfg, test_size=self.settings.test_size,
                                random_state=self.settings.random_state)
        return loaded, prep.prepare(loaded.data), cfg

    def _base_model(self, prepared, params: dict | None = None):
        from lightgbm import LGBMClassifier
        defaults = dict(self.settings.baseline_model)
        defaults.update(params or {})
        defaults["random_state"] = self.settings.random_state
        defaults["verbose"] = -1
        defaults["n_jobs"] = 1
        return LGBMClassifier(**defaults)

    def _sense_series(self, prepared, attr: str) -> pd.Series:
        return prepared.sensitive_test[attr].reset_index(drop=True)

    def _evaluate(self, prepared, y_pred: pd.Series, y_prob, sens: dict) -> dict:
        y_true = prepared.y_test.reset_index(drop=True)
        sens_clean = {k: v.reset_index(drop=True) for k, v in sens.items()}
        fairness = full_fairness_report(y_true, y_pred, sens_clean)
        try:
            if y_prob is not None and np.asarray(y_prob).ndim == 2:
                y_prob = np.asarray(y_prob)[:, 1]
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": auc,
            **fairness,
        }

    def _run_one(self, ds_name: str) -> list[dict]:
        loaded, prepared, cfg = self._load_prepared(ds_name)
        s_train = prepared.sensitive_train[cfg.sensitive_attributes[0]].reset_index(drop=True)
        s_test = self._sense_series(prepared, cfg.sensitive_attributes[0])

        rows: list[dict] = []

        base = self._base_model(prepared)
        base.fit(prepared.X_train, prepared.y_train)
        y_pred_base = pd.Series(base.predict(prepared.X_test), index=prepared.y_test.index)
        y_prob_base = base.predict_proba(prepared.X_test)
        baseline_metrics = self._evaluate(prepared, y_pred_base, y_prob_base,
                                          {attr: self._sense_series(prepared, attr) for attr in cfg.sensitive_attributes})
        rows.append({"dataset": ds_name, "method": "baseline", "lambda": "",
                     **baseline_metrics})

        exp_rows = self._expgrad(prepared, s_train, s_test, ds_name, cfg)
        rows.extend(exp_rows)
        thr_rows = self._threshold_optimizer(prepared, s_train, s_test, ds_name, base)
        rows.extend(thr_rows)

        return rows

    def _require_fairlearn(self) -> None:
        try:
            from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds  # noqa: F401
            from fairlearn.postprocessing import ThresholdOptimizer  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("W3 requires fairlearn. Install requirements.txt before running fairness experiments.") from exc

    def _expgrad(self, prepared, s_train: pd.Series, s_test: pd.Series, ds_name: str, cfg) -> list[dict]:
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

        rows: list[dict] = []
        for constraint_name, ConstraintCls in [("demographic_parity", DemographicParity),
                                                ("equalized_odds", EqualizedOdds)]:
            for lam in self.lambdas:
                try:
                    constraint = ConstraintCls()
                    base = self._base_model(prepared)
                    mitigator = ExponentiatedGradient(
                        estimator=base,
                        constraints=constraint,
                        max_iter=self.settings.fairlearn_config.get("max_iter", 50),
                        eps=lam,
                    )
                    mitigator.fit(prepared.X_train, prepared.y_train, sensitive_features=s_train)
                    y_pred = pd.Series(mitigator.predict(prepared.X_test),
                                        index=prepared.y_test.index)
                    try:
                        y_prob = mitigator._pmf_predict(prepared.X_test)[:, 1]
                    except Exception:
                        y_prob = None
                    metrics = self._evaluate(prepared, y_pred, y_prob,
                                              {attr: self._sense_series(prepared, attr)
                                               for attr in cfg.sensitive_attributes})
                    rows.append({"dataset": ds_name, "method": f"expgrad_{constraint_name}",
                                 "lambda": lam, **metrics})
                except Exception as exc:
                    rows.append({"dataset": ds_name, "method": f"expgrad_{constraint_name}",
                                 "lambda": lam, "error": str(exc)})
        return rows

    def _threshold_optimizer(self, prepared, s_train: pd.Series, s_test: pd.Series, ds_name: str, base) -> list[dict]:
        from fairlearn.postprocessing import ThresholdOptimizer

        rows: list[dict] = []
        try:
            base.fit(prepared.X_train, prepared.y_train)
            y_prob_test = base.predict_proba(prepared.X_test)
            cfg_inner = self.settings.dataset(ds_name)
            sens_test_dict = {attr: self._sense_series(prepared, attr) for attr in cfg_inner.sensitive_attributes}

            # ThresholdOptimizer cannot fit a group whose training labels contain
            # only one class. Fit on eligible groups and use the baseline model for
            # those degenerate groups, recording the fallback explicitly.
            y_train = prepared.y_train.reset_index(drop=True)
            group_stats = pd.DataFrame({"group": s_train.astype(str), "y": y_train})
            group_nunique = group_stats.groupby("group")["y"].nunique()
            eligible_groups = set(group_nunique[group_nunique >= 2].index)
            fallback_groups = sorted(set(s_train.astype(str)) - eligible_groups)
            fit_mask = s_train.astype(str).isin(eligible_groups).to_numpy()
            if not eligible_groups:
                raise ValueError("No sensitive groups have both labels in the training data")
            x_fit = prepared.X_train.iloc[fit_mask]
            y_fit = prepared.y_train.iloc[fit_mask]
            s_fit = s_train.iloc[fit_mask]

            for constraint_name in ["demographic_parity", "equalized_odds"]:
                try:
                    postprocess = ThresholdOptimizer(
                        estimator=base,
                        constraints=constraint_name,
                        objective="balanced_accuracy_score",
                        prefit=True,
                    )
                    postprocess.fit(x_fit, y_fit, sensitive_features=s_fit)

                    y_pred = np.asarray(base.predict(prepared.X_test)).ravel()
                    test_groups = s_test.astype(str)
                    eligible_test_mask = test_groups.isin(eligible_groups).to_numpy()
                    if eligible_test_mask.any():
                        x_test_eligible = prepared.X_test.iloc[eligible_test_mask]
                        s_test_eligible = test_groups.iloc[eligible_test_mask]
                        try:
                            y_post = postprocess.predict(
                                x_test_eligible, sensitive_features=s_test_eligible
                            )
                        except TypeError:
                            y_post = postprocess.predict(x_test_eligible, s_test_eligible)
                        y_pred[eligible_test_mask] = np.asarray(y_post).ravel()

                    metrics = self._evaluate(
                        prepared,
                        pd.Series(y_pred, index=prepared.y_test.index),
                        y_prob_test,
                        sens_test_dict,
                    )
                    rows.append({
                        "dataset": ds_name,
                        "method": f"threshold_{constraint_name}",
                        "lambda": "",
                        "threshold_fallback_groups": ";".join(fallback_groups),
                        **metrics,
                    })
                except Exception as exc:
                    rows.append({
                        "dataset": ds_name,
                        "method": f"threshold_{constraint_name}",
                        "lambda": "",
                        "threshold_fallback_groups": ";".join(fallback_groups),
                        "error": str(exc),
                    })
        except Exception as exc:
            for constraint_name in ["demographic_parity", "equalized_odds"]:
                rows.append({
                    "dataset": ds_name,
                    "method": f"threshold_{constraint_name}",
                    "lambda": "",
                    "error": str(exc),
                })
        return rows
    def _select_pareto(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cand = df.dropna(subset=["accuracy"]).copy()
        if cand.empty:
            return cand
        pareto = []
        for _, group in cand.groupby("dataset", dropna=False):
            fair_cols = [c for c in group.columns if c.endswith("__spd_spd_range")]
            if not fair_cols:
                pareto.extend(group.to_dict("records"))
                continue
            group = group.copy()
            group["__primary_fair"] = group[fair_cols].mean(axis=1, skipna=True).abs()
            group = group.dropna(subset=["__primary_fair"])
            for _, row in group.iterrows():
                dominated = ((group["accuracy"] >= row["accuracy"]) & (group["__primary_fair"] <= row["__primary_fair"]) & ((group["accuracy"] > row["accuracy"]) | (group["__primary_fair"] < row["__primary_fair"]))).any()
                if not dominated:
                    pareto.append(row.drop(labels=["__primary_fair"]).to_dict())
        return pd.DataFrame(pareto)

    def _build_markdown(self, summary: pd.DataFrame, pareto: pd.DataFrame) -> str:
        lines = ["# W3 - Fairness", ""]
        lines.append(f"Datasets: **{summary['dataset'].nunique()}**")
        lines.append(f"Methods: **{summary['method'].nunique()}**")
        lines.append("")
        cols = [c for c in ["dataset", "method", "accuracy", "auc"] if c in summary.columns]
        lines.append("## Model comparison")
        lines.append("")
        if not summary.empty:
            lines.append(summary[cols].head(20).to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append("## Pareto frontier")
        lines.append("")
        if not pareto.empty:
            pcols = [c for c in ["dataset", "method", "accuracy", "auc"] if c in pareto.columns]
            pcols += [c for c in pareto.columns if c.endswith("__spd_spd_range")]
            lines.append(pareto[pcols].to_markdown(index=False, floatfmt=".3f"))
        else:
            lines.append("(empty)")
        return "\n".join(lines) + "\n"



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sbo", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"])
    args = parser.parse_args()
    W3Fairness(lambdas=[0.1, 0.5, 1.0]).run(datasets=args.datasets)
