"""W2 - Accuracy: Optuna tuning + 3-fold CV + model comparison."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from fairness_lab.data.loader import DataLoader
from fairness_lab.data.preprocessing import DataPreprocessor
from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.settings import Settings
from fairness_lab.utils.io import ensure_dir, write_dataframe, write_json, write_markdown


class W2Accuracy:
    """Week 2: hyperparameter tuning + 3-fold CV + comparison with XGBoost/CatBoost."""

    def __init__(self, out_dir: Path | None = None, n_trials: int = 50) -> None:
        self.out_dir = ensure_dir(out_dir or (ARTIFACTS_DIR / "w2"))
        self.n_trials = n_trials
        self.settings = Settings()

    def run(self, datasets: list[str], skip_optuna: bool = False) -> None:
        print(f"[W2] skip_optuna={skip_optuna}, datasets={datasets}")
        rows: list[dict] = []
        optuna_results: dict[str, dict] = {}

        for ds_name in datasets:
            print(f"[W2] -> {ds_name}")
            try:
                result = self._run_one(ds_name, skip_optuna=skip_optuna)
            except Exception as exc:
                print(f"  !! {ds_name} failed: {exc}")
                continue
            rows.append(result)
            optuna_results[ds_name] = {
                "status": result.get("optuna_status", "unknown"),
                "used": bool(result.get("optuna_used", False)),
                "best_value": result.get("optuna_best_value"),
                "best_params": result.get("model_results", {}).get("lgbm", {}).get("params", result.get("best_params", {})),
                "error": result.get("optuna_error", ""),
                "trials": self.n_trials,
            }

        if not rows:
            print("[W2] No datasets produced results.")
            return

        df = pd.DataFrame(rows)
        write_dataframe(df, self.out_dir / "accuracy_summary.csv")
        write_json({r["dataset"]: r for r in rows}, self.out_dir / "per_dataset.json")
        write_json(optuna_results, self.out_dir / "optuna_results.json")
        print(f"[W2] wrote artefacts to {self.out_dir}")

    def _load_prepared(self, ds_name: str):
        cfg = self.settings.dataset(ds_name)
        loader = DataLoader(random_state=self.settings.random_state)
        loaded = loader.load(cfg)
        prep = DataPreprocessor(dataset_cfg=cfg, test_size=self.settings.test_size,
                                random_state=self.settings.random_state)
        return loaded, prep.prepare(loaded.data)

    def _run_one(self, ds_name: str, skip_optuna: bool = False) -> dict:
        loaded, prepared = self._load_prepared(ds_name)

        baseline_params = dict(self.settings.baseline_model)
        best_params = baseline_params.copy()
        optuna_meta = {
            "optuna_status": "skipped_by_flag" if skip_optuna else "not_run",
            "optuna_used": False,
            "optuna_best_value": None,
            "optuna_error": "",
        }

        if not skip_optuna:
            best_params, optuna_meta = self._optuna_tune(
                prepared.X_train, prepared.y_train, baseline_params
            )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.settings.random_state)

        lgbm_row = self._cv_score(
            ds_name, "lgbm_optuna", best_params, prepared, cv,
        )
        lgbm_row["params"] = best_params
        lgbm_row["model"] = "lgbm"

        xgb_row = self._cv_score(
            ds_name, "xgboost", {}, prepared, cv,
        )
        cat_row = self._cv_score(
            ds_name, "catboost", {}, prepared, cv,
        )

        best = max([lgbm_row, xgb_row, cat_row], key=lambda r: r.get("auc_cv", float("-inf")))
        return {
            "dataset": ds_name,
            "source": loaded.source,
            "best_model": best["model"],
            "best_auc_cv": float(best.get("auc_cv", float("nan"))),
            "best_auc_cv_std": float(best.get("auc_cv_std", float("nan"))),
            "best_accuracy": float(best.get("accuracy", float("nan"))),
            "best_f1": float(best.get("f1", float("nan"))),
            "best_auc_test": float(best.get("auc", float("nan"))),
            "best_params": best.get("params", {}),
            "model_results": {r["model"]: r for r in [lgbm_row, xgb_row, cat_row]},
            **optuna_meta,
        }

    def _build_model(self, name: str, params: dict) -> Any:
        if name in ("lgbm", "lgbm_optuna"):
            try:
                from lightgbm import LGBMClassifier
                merged = dict(self.settings.baseline_model)
                merged.update(params)
                merged["random_state"] = self.settings.random_state
                merged["verbose"] = -1
                return LGBMClassifier(**merged)
            except ImportError as exc:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=300, random_state=self.settings.random_state)
        if name == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     random_state=self.settings.random_state, verbosity=0,
                                     use_label_encoder=False, eval_metric="logloss", n_jobs=1)
            except ImportError as exc:
                return None
        if name == "catboost":
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                          random_state=self.settings.random_state, verbose=0, thread_count=1)
            except ImportError as exc:
                return None
        raise ValueError(f"Unknown model: {name}")

    def _eval_model(self, model, X_train, X_test, y_train, y_test) -> dict:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            proba = None
            auc = 0.0
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": auc,
        }

    def _cv_score(self, ds_name: str, model_name: str, extra_params: dict,
                  prepared, cv: StratifiedKFold) -> dict:
        model = self._build_model(model_name, extra_params)
        if model is None:
            return {
                "dataset": ds_name,
                "model": model_name,
                "auc_cv": 0.0,
                "skipped": True,
                "skip_reason": f"{model_name} dependency is not installed",
            }
        try:
            scores = cross_val_score(model, prepared.X_train, prepared.y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=1)
            metrics = self._eval_model(model, prepared.X_train, prepared.X_test,
                                       prepared.y_train, prepared.y_test)
        except Exception as exc:
            print(f"  !! CV {model_name} failed: {exc}")
            return {
                "dataset": ds_name,
                "model": model_name,
                "auc_cv": 0.0,
                "skipped": True,
                "skip_reason": f"{model_name} dependency is not installed",
            }
        return {
            "dataset": ds_name,
            "model": model_name,
            "auc_cv": float(scores.mean()),
            "auc_cv_std": float(scores.std()),
            **metrics,
        }

    def _optuna_tune(self, X_train, y_train, base_params: dict) -> tuple[dict, dict]:
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            return base_params, {
                "optuna_status": "dependency_missing",
                "optuna_used": False,
                "optuna_best_value": None,
                "optuna_error": str(exc),
            }

        search_space = self.settings.optuna_search_space

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators",
                                                   int(search_space.get("n_estimators", [200, 1000])[0]),
                                                   int(search_space.get("n_estimators", [200, 1000])[1])),
                "learning_rate": trial.suggest_float("learning_rate",
                                                     *search_space.get("learning_rate", [0.01, 0.3]),
                                                     log=True),
                "num_leaves": trial.suggest_int("num_leaves",
                                                int(search_space.get("num_leaves", [15, 127])[0]),
                                                int(search_space.get("num_leaves", [15, 127])[1])),
                "max_depth": trial.suggest_int("max_depth",
                                               int(search_space.get("max_depth", [3, 12])[0]),
                                               int(search_space.get("max_depth", [3, 12])[1])),
                "subsample": trial.suggest_float("subsample",
                                                 *search_space.get("subsample", [0.6, 1.0])),
                "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                         *search_space.get("colsample_bytree", [0.6, 1.0])),
                "random_state": self.settings.random_state,
                "verbose": -1,
            }
            model = LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.settings.random_state)
            try:
                return float(cross_val_score(model, X_train, y_train, cv=cv,
                                             scoring="roc_auc", n_jobs=1).mean())
            except Exception:
                return 0.0

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.settings.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        result = dict(base_params)
        result.update(study.best_params)
        print(f"  [optuna] best_auc={study.best_value:.4f}  params={study.best_params}")
        return result, {
            "optuna_status": "completed",
            "optuna_used": True,
            "optuna_best_value": float(study.best_value),
            "optuna_error": "",
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sbo", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"])
    parser.add_argument("--skip-optuna", action="store_true")
    args = parser.parse_args()
    W2Accuracy(n_trials=50).run(datasets=args.datasets, skip_optuna=args.skip_optuna)
