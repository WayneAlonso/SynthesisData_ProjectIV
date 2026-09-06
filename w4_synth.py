"""W4 - Synthetic data usability: 4 synthesizers + sdmetrics + TSTR + DP noise."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from fairness_lab.data.loader import DataLoader
from fairness_lab.data.preprocessing import DataPreprocessor
from fairness_lab.fairness.metrics import full_fairness_report
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.settings import Settings
from fairness_lab.utils.io import ensure_dir, write_dataframe, write_json, write_markdown


class W4Synth:
    """Week 4: synthetic-data generation + evaluation + TSTR + DP proxy."""

    def __init__(
        self,
        out_dir: Path | None = None,
        n_synth: list[str] | None = None,
        privacy_levels: list[float] | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        self.out_dir = ensure_dir(out_dir or (ARTIFACTS_DIR / "w4"))
        self.n_synth = n_synth or ["ctgan", "tvae", "gaussian_copula", "copula_gan"]
        self.privacy_levels = privacy_levels or [0.0, 0.1, 0.3]
        self.settings = Settings()
        self.use_gpu = use_gpu if use_gpu is not None else __import__("os").environ.get("W4_USE_GPU", "0") == "1"

    def run(self, datasets: list[str]) -> None:
        if self.use_gpu:
            import os
            os.environ["W4_USE_GPU"] = "1"
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("W4_USE_GPU=1 but PyTorch CUDA is unavailable")
                torch.cuda.set_device(0)
                print(f"[W4] PyTorch CUDA: {torch.cuda.get_device_name(0)} | VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            except ImportError as exc:
                raise RuntimeError("GPU mode requires PyTorch in the fns environment") from exc
        print(f"[W4] datasets={datasets}, synth={self.n_synth}, dp={self.privacy_levels}")
        rows: list[dict] = []
        for ds_name in datasets:
            print(f"[W4] -> {ds_name}")
            try:
                rows.extend(self._run_one(ds_name))
                # Checkpoint after every completed dataset so long W4 runs are resumable.
                write_dataframe(pd.DataFrame(rows), self.out_dir / "synth_evaluation_checkpoint.csv")
            except Exception as exc:
                print(f"  !! {ds_name} failed: {exc}")

        if not rows:
            print("[W4] No dataset produced results.")
            return

        df = pd.DataFrame(rows)
        write_dataframe(df, self.out_dir / "synth_evaluation.csv")
        report = self._build_markdown(df)
        write_markdown(report, self.out_dir / "W4_SYNTH.md")
        print(f"[W4] wrote artefacts to {self.out_dir}")

    def _load_prepared(self, ds_name: str):
        cfg = self.settings.dataset(ds_name)
        loader = DataLoader(random_state=self.settings.random_state)
        loaded = loader.load(cfg)
        prep = DataPreprocessor(dataset_cfg=cfg, test_size=self.settings.test_size,
                                random_state=self.settings.random_state)
        return loaded, prep.prepare(loaded.data), cfg

    def _synthesizer(self, name: str, params: dict | None = None) -> Any:
        from fairness_lab.models.generative import SDVTableSynthesizer
        cfg = dict((params or self.settings.synthesis_models_config).get(name, {}))
        return SDVTableSynthesizer(model_name=name, params=cfg, random_state=self.settings.random_state)

    def _apply_dp_noise(
        self,
        df: pd.DataFrame,
        level: float,
        rng: np.random.Generator,
        protected_columns: set[str] | None = None,
        continuous_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Apply a clearly-labelled numeric noise proxy without corrupting labels/categories.

        This is not differential privacy. It is retained only as a sensitivity analysis
        and never modifies the target, sensitive attributes, or categorical columns.
        """
        if level <= 0.0:
            return df.copy()
        out = df.copy()
        protected = protected_columns or set()
        num_cols = continuous_columns if continuous_columns is not None else out.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in protected:
                continue
            std = out[col].std()
            if std and not pd.isna(std):
                out[col] = out[col] + rng.normal(0.0, level * std, size=len(out))
        return out

    def _sdmetrics_quality(self, real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            from sdmetrics.single_table import NewRowSynthesis
            try:
                score = NewRowSynthesis.compute(real, synth, synthetic_sample_size=min(200, len(synth)))
                metrics["new_row_synthesis"] = float(score)
            except Exception as exc:
                metrics["new_row_synthesis"] = self._new_row_synthesis_proxy(real, synth)
                metrics["quality_error_new_row_synthesis"] = str(exc)
                metrics["quality_backend"] = "proxy_fallback"

            try:
                from sdmetrics.column_pairs import CorrelationSimilarity
                score = CorrelationSimilarity.compute(real, synth.head(min(200, len(synth))))
                metrics["correlation_similarity"] = float(score)
            except Exception as exc:
                metrics["correlation_similarity"] = self._correlation_similarity_proxy(real, synth)
                metrics["quality_error_correlation_similarity"] = str(exc)
                metrics["quality_backend"] = "proxy_fallback"
        except Exception as exc:
            metrics["new_row_synthesis"] = self._new_row_synthesis_proxy(real, synth)
            metrics["correlation_similarity"] = self._correlation_similarity_proxy(real, synth)
            metrics["quality_backend"] = "proxy_fallback"
            metrics["quality_error"] = str(exc)

        try:
            num_real = real.select_dtypes(include=[np.number]).columns.tolist()
            num_synth = synth.select_dtypes(include=[np.number]).columns.tolist()
            common = list(set(num_real) & set(num_synth))
            if common:
                from scipy.stats import ks_2samp
                ks_vals: list[float] = []
                for col in common[:5]:
                    try:
                        stat, _ = ks_2samp(real[col].dropna(), synth[col].dropna())
                        ks_vals.append(float(stat))
                    except Exception:
                        pass
                if ks_vals:
                    metrics["ks_mean"] = float(np.mean(ks_vals))
        except Exception:
            pass
        metrics.setdefault("quality_backend", "sdmetrics")
        return metrics

    @staticmethod
    def _new_row_synthesis_proxy(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        """Fallback novelty score used only when SDMetrics cannot evaluate the table."""
        if synth.empty:
            return float("nan")
        cols = [c for c in real.columns if c in synth.columns]
        if not cols:
            return float("nan")
        sample = synth[cols].head(min(2000, len(synth))).astype("string").fillna("<NA>")
        real_keys = set(real[cols].astype("string").fillna("<NA>").itertuples(index=False, name=None))
        novel = [tuple(row) not in real_keys for row in sample.itertuples(index=False, name=None)]
        return float(np.mean(novel))

    @staticmethod
    def _correlation_similarity_proxy(real: pd.DataFrame, synth: pd.DataFrame) -> float:
        cols = [c for c in real.select_dtypes(include=[np.number]).columns if c in synth.columns]
        if len(cols) < 2:
            return float("nan")
        real_corr = real[cols].corr().fillna(0.0).to_numpy()
        synth_corr = synth[cols].corr().fillna(0.0).to_numpy()
        return float(np.clip(1.0 - np.mean(np.abs(real_corr - synth_corr)), 0.0, 1.0))

    def _tstr(self, synth_df: pd.DataFrame, prepared, cfg) -> dict:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            LGBMClassifier = None  # noqa: N806

        try:
            synth_train = prepared.train_df.copy()
            target = cfg.target_column
            sens = [c for c in cfg.sensitive_attributes if c in synth_train.columns]
            cat_cols = [c for c in cfg.categorical_columns
                        if c in prepared.feature_columns and c in synth_train.columns and c != target and c not in sens]
            num_cols = [c for c in cfg.numerical_columns
                        if c in prepared.feature_columns and c in synth_train.columns and c != target and c not in sens]
            feats = [c for c in (cat_cols + num_cols) if c in synth_df.columns and c in synth_train.columns]
            if not feats or target not in synth_df.columns:
                return {"tstr_accuracy": float("nan")}

            x_synth = synth_df[feats].copy()
            y_synth = synth_df[target].astype(int)

            params = dict(self.settings.baseline_model)
            params.update({"random_state": self.settings.random_state, "verbose": -1, "n_jobs": 1})
            if self.use_gpu:
                params["device_type"] = "gpu"
                params.setdefault("n_jobs", 1)
            if LGBMClassifier is not None:
                model = LGBMClassifier(**params)
            else:
                model = RandomForestClassifier(n_estimators=200, random_state=self.settings.random_state)

            from sklearn.preprocessing import OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            num_in = [c for c in num_cols if c in feats]
            cat_in = [c for c in cat_cols if c in feats]
            ct = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_in),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                   ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_in),
            ])
            pipe = Pipeline([("ct", ct), ("m", model)])
            pipe.fit(x_synth, y_synth)

            x_test_real = prepared.test_df[feats].copy()
            y_test_real = prepared.test_df[target].astype(int)
            y_pred = pipe.predict(x_test_real)
            try:
                proba = pipe.predict_proba(x_test_real)[:, 1]
                auc = float(roc_auc_score(y_test_real, proba))
            except Exception:
                proba = None
                auc = 0.0
            sens_test = {a: prepared.sensitive_test[a] for a in cfg.sensitive_attributes if a in prepared.sensitive_test}
            fairness = full_fairness_report(y_test_real.reset_index(drop=True),
                                             pd.Series(y_pred).reset_index(drop=True),
                                             {k: v.reset_index(drop=True) for k, v in sens_test.items()})
            return {
                "tstr_accuracy": float(accuracy_score(y_test_real, y_pred)),
                "tstr_f1": float(f1_score(y_test_real, y_pred, zero_division=0)),
                "tstr_auc": float(auc),
                **{f"tstr_{k}": v for k, v in fairness.items()},
            }
        except Exception as exc:
            return {"tstr_accuracy": float("nan"), "tstr_error": str(exc)}

    def _run_one(self, ds_name: str) -> list[dict]:
        loaded, prepared, cfg = self._load_prepared(ds_name)
        rng = np.random.default_rng(self.settings.random_state)
        real_train = prepared.train_df.copy()
        n_train = len(real_train)
        rows: list[dict] = []
        synthesis_cache: dict[str, Any] = {}

        for name in self.n_synth:
            try:
                # Train each synthesizer once. Privacy levels are post-generation
                # sensitivity analyses and must not retrain the same generator.
                if name not in synthesis_cache:
                    synthesis_cache[name] = self._synthesizer(name).run(real_train, num_rows=n_train)
                result = synthesis_cache[name]
            except Exception as exc:
                for level in self.privacy_levels:
                    rows.append({"dataset": ds_name, "synthesizer": name,
                                 "privacy_level": level, "error": str(exc)})
                continue

            for level in self.privacy_levels:
                try:
                    synth_df = result.synthetic_data.copy()
                    protected = {cfg.target_column, *cfg.sensitive_attributes, *cfg.categorical_columns}
                    continuous = [c for c in cfg.numerical_columns
                                  if c in synth_df.columns and c not in protected]
                    synth_df = self._apply_dp_noise(
                        synth_df, level, rng,
                        protected_columns=protected,
                        continuous_columns=continuous,
                    )
                    synth_df = synth_df.reindex(columns=prepared.train_df.columns)
                    quality_train = self._sdmetrics_quality(prepared.train_df, synth_df)
                    quality_test = self._sdmetrics_quality(prepared.test_df, synth_df)
                    quality = {f"train_{k}": v for k, v in quality_train.items()}
                    quality.update({f"test_{k}": v for k, v in quality_test.items()})
                    tstr = self._tstr(synth_df, prepared, cfg)

                    trtr_acc = float("nan")
                    try:
                        from lightgbm import LGBMClassifier
                        from sklearn.preprocessing import OneHotEncoder
                        from sklearn.compose import ColumnTransformer
                        from sklearn.pipeline import Pipeline
                        from sklearn.impute import SimpleImputer
                        params = dict(self.settings.baseline_model)
                        params.update({"random_state": self.settings.random_state, "verbose": -1, "n_jobs": 1})
                        if self.use_gpu:
                            params["device_type"] = "gpu"
                            params.setdefault("n_jobs", 1)
                        feats = [c for c in prepared.feature_columns
                                 if c in prepared.train_df.columns and c != cfg.target_column]
                        ct = ColumnTransformer([
                            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]),
                             [c for c in cfg.numerical_columns if c in feats]),
                            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
                             [c for c in cfg.categorical_columns if c in feats]),
                        ])
                        trtr_pipe = Pipeline([("ct", ct), ("m", LGBMClassifier(**params))])
                        trtr_pipe.fit(synth_df[feats], synth_df[cfg.target_column].astype(int))
                        pred = trtr_pipe.predict(prepared.test_df[feats])
                        trtr_acc = float(accuracy_score(prepared.test_df[cfg.target_column].astype(int), pred))
                    except Exception:
                        pass

                    rows.append({
                        "dataset": ds_name,
                        "synthesizer": name,
                        "backend": result.backend,
                        "privacy_level": level,
                        "n_synth": int(len(synth_df)),
                        "trtr_accuracy": trtr_acc,
                        **quality,
                        **tstr,
                    })
                except Exception as exc:
                    rows.append({"dataset": ds_name, "synthesizer": name,
                                 "privacy_level": level, "error": str(exc)})
        return rows


    def _build_markdown(self, df: pd.DataFrame) -> str:
        lines: list[str] = ["# W4 - Synthetic Data", ""]
        lines.append(f"结果行数：**{len(df)}**")
        lines.append("")
        cols = ["dataset", "synthesizer", "backend", "privacy_level", "trtr_accuracy",
                "tstr_accuracy", "tstr_auc"]
        cols = [c for c in cols if c in df.columns]
        if not df.empty:
            lines.append(df[cols].head(20).to_markdown(index=False, floatfmt=".3f"))
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sbo", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"])
    args = parser.parse_args()
    W4Synth(
        n_synth=["ctgan", "tvae", "gaussian_copula", "copula_gan"],
        privacy_levels=[0.0, 0.1, 0.3],
    ).run(datasets=args.datasets)
