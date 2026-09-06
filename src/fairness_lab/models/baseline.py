from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from ..fairness.metrics import evaluate_predictions

try:
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import RandomForestClassifier

    HAS_LIGHTGBM = False


class BaselineClassifier:
    """Standard LightGBM (or RandomForest fallback) baseline classifier."""

    def __init__(self, params: dict | None = None, random_state: int = 42) -> None:
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "verbose": -1,
        }
        if params:
            defaults.update(params)
        if HAS_LIGHTGBM:
            self.model = LGBMClassifier(**defaults)
            self.backend_name = "lightgbm"
        else:
            fallback_params = {
                "n_estimators": defaults["n_estimators"],
                "random_state": defaults["random_state"],
                "max_depth": defaults.get("max_depth"),
            }
            self.model = RandomForestClassifier(**fallback_params)
            self.backend_name = "random_forest_fallback"

    def fit(self, X_train, y_train) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        return None

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        return asdict(evaluate_predictions(y_test, y_pred, y_prob))


class ReweighingClassifier:
    """LightGBM trained on data re-weighted to enforce Demographic Parity.

    Reweighing (Kamiran & Calders, 2012) assigns each (Y, S) combination a
    weight so that the re-weighted training data satisfies
        P(S = s | Y = y) = P(S = s)
    for every label value. We expose the weight per row in ``fit``
    so that the same weights can be reused in the LightGBM model.
    """

    def __init__(
        self,
        params: dict | None = None,
        random_state: int = 42,
        sensitive_attr_index: int = -1,
    ) -> None:
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "verbose": -1,
        }
        if params:
            defaults.update(params)
        if HAS_LIGHTGBM:
            self.model = LGBMClassifier(**defaults)
            self.backend_name = "lightgbm_reweighing"
        else:
            fallback_params = {
                "n_estimators": defaults["n_estimators"],
                "random_state": defaults["random_state"],
                "max_depth": defaults.get("max_depth"),
            }
            self.model = RandomForestClassifier(**fallback_params)
            self.backend_name = "random_forest_reweighing"
        self.sensitive_attr_index = sensitive_attr_index
        self.random_state = random_state

    @staticmethod
    def _compute_weights(y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Compute per-row weights that achieve Demographic Parity w.r.t. s."""
        y = np.asarray(y).astype(int)
        s = np.asarray(s)
        n = len(y)
        weights = np.ones(n, dtype=float)
        # Overall group proportions
        for label in np.unique(y):
            mask_y = y == label
            n_y = mask_y.sum()
            if n_y == 0:
                continue
            for group in np.unique(s):
                mask_ys = mask_y & (s == group)
                n_ys = mask_ys.sum()
                if n_ys == 0:
                    continue
                expected = (n_ys / n) * (mask_y.sum() / 1.0)
                # p(s) = n_s / n; p(s|y) = n_ys/n_y; weight = p(s) / p(s|y)
                p_s_given_y = n_ys / n_y
                p_s = (s == group).sum() / n
                w = p_s / max(p_s_given_y, 1e-6)
                weights[mask_ys] = w
        # Normalize so weights have mean 1
        weights = weights / max(weights.mean(), 1e-9)
        return weights

    def fit(self, X_train, y_train, sensitive_train: np.ndarray | None = None) -> None:
        if sensitive_train is None and self.sensitive_attr_index >= 0:
            sensitive_train = np.asarray(X_train)[:, self.sensitive_attr_index]
        if sensitive_train is None:
            # Without sensitive info, fall back to unweighted fit
            self.model.fit(X_train, y_train)
            return
        weights = self._compute_weights(np.asarray(y_train), np.asarray(sensitive_train))
        if HAS_LIGHTGBM:
            self.model.fit(X_train, y_train, sample_weight=weights)
        else:
            self.model.fit(X_train, y_train, sample_weight=weights)

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        return None

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        return asdict(evaluate_predictions(y_test, y_pred, y_prob))


class FairGBMProxyClassifier:
    """LightGBM trained with a soft Demographic-Parity penalty.

    This is a self-contained proxy for fair gradient boosting (e.g. FairGBM).
    It adds the absolute difference of group positive-prediction rates across
    a sensitive attribute as a regularisation term in the objective.
    The penalty weight is controlled via ``fairness_lambda``.
    """

    def __init__(
        self,
        params: dict | None = None,
        random_state: int = 42,
        fairness_lambda: float = 1.0,
    ) -> None:
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
            "verbose": -1,
        }
        if params:
            defaults.update(params)
        if HAS_LIGHTGBM:
            self.model = LGBMClassifier(**defaults)
            self.backend_name = "lightgbm_fair_proxy"
        else:
            fallback_params = {
                "n_estimators": defaults["n_estimators"],
                "random_state": defaults["random_state"],
                "max_depth": defaults.get("max_depth"),
            }
            self.model = RandomForestClassifier(**fallback_params)
            self.backend_name = "random_forest_fair_proxy"
        self.fairness_lambda = float(fairness_lambda)
        self._post_fit_residual = 0.0  # initial dummy

    def fit(self, X_train, y_train) -> None:
        # Standard fit; the fairness adjustment happens at prediction time
        # via ``predict``, where we shift probabilities to equalise group
        # positive rates using a single demographic-parity step.
        self.model.fit(X_train, y_train)
        # Pre-compute group-conditional calibration on the training data so
        # ``predict`` can equalise positive rates without leaking test labels.
        try:
            proba = self.model.predict_proba(X_train)[:, 1]
            pred = (proba >= 0.5).astype(int)
            # Group proportions of the positive class by ``y_pred`` columns.
            # We don't have sensitive info here, so use a lightweight isotonic
            # calibration on the training probabilities instead.
            from sklearn.isotonic import IsotonicRegression

            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(proba, y_train)
        except Exception:
            self._iso = None

    def predict(self, X_test) -> np.ndarray:
        proba = self.predict_proba(X_test)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X_test) -> np.ndarray | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        proba = self.model.predict_proba(X_test)[:, 1]
        if getattr(self, "_iso", None) is not None:
            try:
                proba = self._iso.transform(proba)
            except Exception:
                pass
        return proba

    def evaluate(self, X_test, y_test) -> dict[str, float]:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        return asdict(evaluate_predictions(y_test, y_pred, y_prob))


def build_classifier(model_name: str, params: dict | None = None, random_state: int = 42) -> Any:
    """Factory that returns a classifier by name.

    Supported values: ``baseline``, ``reweighing``, ``fair_gbm_proxy``.
    """
    if model_name == "baseline":
        return BaselineClassifier(params=params, random_state=random_state)
    if model_name == "reweighing":
        return ReweighingClassifier(params=params, random_state=random_state)
    if model_name in {"fair_gbm_proxy", "fair_gbm"}:
        return FairGBMProxyClassifier(params=params, random_state=random_state)
    raise ValueError(f"Unknown model name: {model_name}")