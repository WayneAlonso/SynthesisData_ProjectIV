"""W1 - Diagnosis & baseline: fairness metrics + OOD audit + pain-point table."""
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


class W1Diagnosis:
    """Week 1: diagnose fairness for baseline LGBM on each dataset."""

    def __init__(self, out_dir: Path | None = None) -> None:
        settings = Settings()
        self.out_dir = ensure_dir(out_dir or (ARTIFACTS_DIR / "w1"))
        self.settings = settings

    def run(self, datasets: list[str]) -> None:
        print(f"[W1] Datasets: {datasets}")
        summary_rows: list[dict] = []
        per_dataset: dict[str, dict] = {}

        for ds_name in datasets:
            print(f"[W1] -> {ds_name}")
            try:
                if ds_name == "sbo_withheld" and "sbo" in datasets:
                    result = self._diagnose_ood("sbo", ds_name)
                else:
                    result = self._diagnose_one(ds_name)
            except Exception as exc:
                print(f"  !! {ds_name} failed: {exc}")
                continue
            summary_rows.append(result["row"])
            per_dataset[ds_name] = result

        if not summary_rows:
            print("[W1] No datasets produced results.")
            return

        df = pd.DataFrame(summary_rows)
        write_dataframe(df, self.out_dir / "fairness_summary.csv")
        write_json(per_dataset, self.out_dir / "per_dataset.json")

        ood_rows = [r for r in summary_rows if r.get("role") == "holdout_audit"]
        if ood_rows:
            write_dataframe(pd.DataFrame(ood_rows), self.out_dir / "ood_audit.csv")
        target_rows = [r for r in summary_rows if r.get("dataset") == "sbo"]
        drift_rows: list[dict] = []
        if target_rows and ood_rows:
            target = target_rows[0]
            for holdout in ood_rows:
                for attr in self._attrs_from_row(holdout):
                    for metric in ("spd_spd_range", "eod_eod_tpr_range", "eod_eod_fpr_range", "di_disparate_impact"):
                        col = f"{attr}__{metric}"
                        if col in target and col in holdout:
                            drift_rows.append({
                                "target_dataset": target["dataset"],
                                "holdout_dataset": holdout["dataset"],
                                "sensitive_attr": attr,
                                "metric": metric,
                                "target_value": float(target[col]),
                                "holdout_value": float(holdout[col]),
                                "drift": float(holdout[col] - target[col]),
                            })
        if drift_rows:
            write_dataframe(pd.DataFrame(drift_rows), self.out_dir / "ood_drift.csv")

        pain = self._build_pain_point_table(df)
        write_dataframe(pain, self.out_dir / "pain_points.csv")

        report = self._build_markdown(df, pain)
        write_markdown(report, self.out_dir / "W1_DIAGNOSIS.md")

        print(f"[W1] wrote artefacts to {self.out_dir}")

    def _diagnose_ood(self, target_name: str, holdout_name: str) -> dict:
        """Train only on target data and evaluate on the untouched holdout domain."""
        target_cfg = self.settings.dataset(target_name)
        holdout_cfg = self.settings.dataset(holdout_name)
        loader = DataLoader(random_state=self.settings.random_state)
        target_loaded = loader.load(target_cfg)
        holdout_loaded = loader.load(holdout_cfg)
        prep = DataPreprocessor(
            dataset_cfg=target_cfg,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
        )
        target = prep.prepare(target_loaded.data)
        holdout = holdout_loaded.data.drop_duplicates().reset_index(drop=True)
        missing = [c for c in target.feature_columns if c not in holdout.columns]
        if missing:
            raise KeyError(f"OOD holdout is missing feature columns: {missing}")
        x_holdout = holdout[target.feature_columns].copy()
        x_holdout = target.transformer.transform(x_holdout)
        if hasattr(x_holdout, "toarray"):
            x_holdout = x_holdout.toarray()
        x_holdout = pd.DataFrame(x_holdout, columns=target.transformer.get_feature_names_out())
        y_holdout = holdout[holdout_cfg.target_column].astype(int).reset_index(drop=True)
        from fairness_lab.models.baseline import BaselineClassifier
        clf = BaselineClassifier(params=dict(self.settings.baseline_model), random_state=self.settings.random_state)
        clf.fit(target.X_train, target.y_train)
        y_pred = pd.Series(clf.predict(x_holdout))
        y_prob = clf.predict_proba(x_holdout)
        sens = {attr: holdout[attr].reset_index(drop=True) for attr in holdout_cfg.sensitive_attributes}
        fairness = full_fairness_report(y_holdout, y_pred, sens)
        metrics = {
            "accuracy": float(accuracy_score(y_holdout, y_pred)),
            "f1": float(f1_score(y_holdout, y_pred, zero_division=0)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(y_holdout, y_prob))
        except Exception:
            metrics["auc"] = float("nan")
        row = {
            "dataset": holdout_name,
            "source": holdout_loaded.source,
            "n_train": int(len(target.X_train)),
            "n_test": int(len(holdout)),
            "positive_rate": float(y_holdout.mean()),
            "role": "holdout_audit",
            "evaluation_protocol": f"train_on_{target_name}_test_on_{holdout_name}",
            "backend": clf.backend_name,
            **metrics,
            **fairness,
        }
        return {"row": row, "metrics": metrics, "fairness": fairness}

    def _diagnose_one(self, ds_name: str) -> dict:
        cfg = self.settings.dataset(ds_name)
        loader = DataLoader(random_state=self.settings.random_state)
        loaded = loader.load(cfg)
        df = loaded.data.copy()

        prep = DataPreprocessor(
            dataset_cfg=cfg,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
        )
        prepared = prep.prepare(df)

        from fairness_lab.models.baseline import BaselineClassifier

        baseline_params = dict(self.settings.baseline_model)
        clf = BaselineClassifier(params=baseline_params, random_state=self.settings.random_state)
        clf.fit(prepared.X_train, prepared.y_train)
        y_pred = pd.Series(clf.predict(prepared.X_test), index=prepared.y_test.index)
        y_prob = clf.predict_proba(prepared.X_test)

        sens = {attr: series for attr, series in prepared.sensitive_test.items() if attr in df.columns}
        fairness = full_fairness_report(prepared.y_test, y_pred, sens)
        metrics = {
            "accuracy": float(accuracy_score(prepared.y_test, y_pred)),
            "f1": float(f1_score(prepared.y_test, y_pred, zero_division=0)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(prepared.y_test, y_prob))
        except Exception:
            metrics["auc"] = float("nan")

        row = {
            "dataset": ds_name,
            "source": loaded.source,
            "n_train": int(len(prepared.X_train)),
            "n_test": int(len(prepared.X_test)),
            "positive_rate": float(prepared.y_test.mean()),
            "role": cfg.role,
            "backend": clf.backend_name,
            **metrics,
            **fairness,
        }
        return {"row": row, "metrics": metrics, "fairness": fairness}

    def _attrs_from_row(self, row: dict | pd.Series) -> list[str]:
        suffix = "__spd_spd_range"
        columns = row.index if hasattr(row, "index") else row.keys()
        return [str(col)[:-len(suffix)] for col in columns if str(col).endswith(suffix)]

    def _build_pain_point_table(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for _, row in df.iterrows():
            ds = row["dataset"]
            attrs = self._attrs_from_row(row)
            for attr in attrs:
                prefix_spd = f"{attr}__spd_spd_range"
                prefix_eod = f"{attr}__eod_eod_tpr_range"
                prefix_di = f"{attr}__di_disparate_impact"
                spd = float(row.get(prefix_spd, 0.0) or 0.0)
                eod = float(row.get(prefix_eod, 0.0) or 0.0)
                di = float(row.get(prefix_di, 1.0) or 1.0)
                severity = "ok"
                if spd > 0.2 or eod > 0.2 or di < 0.8:
                    severity = "high"
                elif spd > 0.1 or eod > 0.1 or di < 0.9:
                    severity = "medium"
                rows.append({
                    "dataset": ds,
                    "sensitive_attr": attr,
                    "spd_range": spd,
                    "eod_tpr_range": eod,
                    "disparate_impact": di,
                    "accuracy": float(row.get("accuracy", 0.0)),
                    "severity": severity,
                })
        return pd.DataFrame(rows)

    def _build_markdown(self, summary: pd.DataFrame, pain: pd.DataFrame) -> str:
        lines: list[str] = ["# W1 - Diagnosis & Baseline", ""]
        lines.append(f"数据集数: **{len(summary)}**")
        lines.append("")
        cols = ["dataset", "n_test", "accuracy", "f1", "auc", "source", "role"]
        cols = [c for c in cols if c in summary.columns]
        lines.append("## 1. 基线指标")
        lines.append("")
        lines.append(summary[cols].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append("## 2. 痛点表")
        lines.append("")
        if not pain.empty:
            lines.append(pain.to_markdown(index=False, floatfmt=".3f"))
        else:
            lines.append("(无)")
        lines.append("")
        lines.append("## 3. 结论")
        lines.append("")
        high = pain[pain["severity"] == "high"] if not pain.empty else pain
        if not high.empty:
            lines.append(f"- 发现 **{len(high)}** 个高严重度痛点组，建议进入 W3 修复")
        else:
            lines.append("- 无高严重度痛点")
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sbo", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"])
    args = parser.parse_args()
    W1Diagnosis().run(datasets=args.datasets)
