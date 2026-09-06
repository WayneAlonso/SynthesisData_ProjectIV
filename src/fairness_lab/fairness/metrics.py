from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _safe_rate(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _group_rates(y_pred: pd.Series, sensitive: pd.Series) -> dict[str, float]:
    rates: dict[str, float] = {}
    for group in sensitive.dropna().unique():
        mask = sensitive == group
        rates[str(group)] = _safe_rate(y_pred[mask])
    return rates


def _pairwise_diff(rates: dict[str, float]) -> dict[str, float]:
    """Compute pairwise differences between group rates.

    Returns both signed differences and the max absolute pairwise difference.
    """
    valid = {k: v for k, v in rates.items() if np.isfinite(v)}
    if len(valid) < 2:
        return {"max_pair_diff": float("nan")}
    values = list(valid.values())
    pairs = {
        f"diff_{a}_minus_{b}": valid[a] - valid[b]
        for a, b in combinations(valid.keys(), 2)
    }
    if pairs:
        pairs["max_pair_diff"] = float(max(abs(v) for v in pairs.values()))
    else:
        pairs["max_pair_diff"] = 0.0
    return pairs


def statistical_parity_difference(y_pred: pd.Series, sensitive: pd.Series) -> dict[str, float]:
    """Statistical Parity Difference (SPD).

    For each group g, returns P(Y_pred = 1 | G = g). The classical SPD is the
    difference between the maximum and minimum group rates; we additionally
    expose the per-group positive rates and pairwise differences so users can
    inspect individual group gaps in the thesis.
    """
    valid = sensitive.notna() & y_pred.notna()
    y_pred = y_pred[valid]
    sensitive = sensitive[valid]
    rates = _group_rates(y_pred, sensitive)
    if not rates:
        return {"spd_range": 0.0}
    valid_rates = [r for r in rates.values() if np.isfinite(r)]
    if not valid_rates:
        return {"spd_range": float("nan")}
    out: dict[str, float] = {
        "spd_range": float(max(valid_rates) - min(valid_rates)),
        **{f"spd_rate_{group}": rate for group, rate in rates.items()},
    }
    out.update({f"spd_{k}": v for k, v in _pairwise_diff(rates).items()})
    return out


def equal_opportunity_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive: pd.Series,
) -> dict[str, float]:
    """Equal Opportunity Difference (EOD) following Hardt et al. (2016).

    Computes per-group true positive rate (TPR) and false positive rate (FPR),
    plus the range and the maximum absolute pairwise difference for both.
    """
    tpr_by_group: dict[str, float] = {}
    fpr_by_group: dict[str, float] = {}
    for group in sensitive.dropna().unique():
        mask_group = sensitive == group
        y_true_g = y_true[mask_group]
        y_pred_g = y_pred[mask_group]
        positives = y_true_g == 1
        negatives = y_true_g == 0
        tpr_by_group[str(group)] = _safe_rate(y_pred_g[positives])
        fpr_by_group[str(group)] = _safe_rate(y_pred_g[negatives])
    if not tpr_by_group:
        return {"eod_tpr_range": 0.0, "eod_fpr_range": 0.0}
    valid_tpr = [r for r in tpr_by_group.values() if np.isfinite(r)]
    valid_fpr = [r for r in fpr_by_group.values() if np.isfinite(r)]
    if not valid_tpr and not valid_fpr:
        return {"eod_tpr_range": float("nan"), "eod_fpr_range": float("nan")}
    out: dict[str, float] = {
        "eod_tpr_range": float(max(valid_tpr) - min(valid_tpr)) if len(valid_tpr) >= 2 else float("nan"),
        "eod_fpr_range": float(max(valid_fpr) - min(valid_fpr)) if len(valid_fpr) >= 2 else float("nan"),
        **{f"eod_tpr_{group}": rate for group, rate in tpr_by_group.items()},
        **{f"eod_fpr_{group}": rate for group, rate in fpr_by_group.items()},
    }
    out.update({f"eod_tpr_{k}": v for k, v in _pairwise_diff(tpr_by_group).items()})
    out.update({f"eod_fpr_{k}": v for k, v in _pairwise_diff(fpr_by_group).items()})
    return out


def disparate_impact(y_pred: pd.Series, sensitive: pd.Series) -> dict[str, float]:
    """Disparate Impact (DI): min P(Y=1|g) / max P(Y=1|g).

    DI = 1 means perfect parity, DI < 1 (often the 4/5 rule) indicates
    disparate impact.
    """
    valid = sensitive.notna() & y_pred.notna()
    y_pred = y_pred[valid]
    sensitive = sensitive[valid]
    rates = _group_rates(y_pred, sensitive)
    if not rates:
        return {"disparate_impact": 1.0}
    max_rate = max(rates.values())
    if max_rate <= 0:
        return {"disparate_impact": 1.0, **rates}
    # Keep zero-rate groups in the numerator; dropping them hides severe disparity.
    di = float(min(rates.values()) / max_rate)
    return {"disparate_impact": di, **rates}


def theil_index(y_pred: pd.Series, sensitive: pd.Series) -> dict[str, float]:
    """Theil Index for the positive-prediction rate across groups.

    T = sum_g (n_g / n) * (r_g / r_bar) * ln(r_g / r_bar)
    where r_g is the positive-prediction rate in group g and r_bar is the
    overall rate. T = 0 indicates perfect group-level parity.
    """
    valid = sensitive.notna() & y_pred.notna()
    y_pred = y_pred[valid]
    sensitive = sensitive[valid]
    rates = _group_rates(y_pred, sensitive)
    if not rates:
        return {"theil_index": 0.0}
    overall = _safe_rate(y_pred)
    if overall <= 0:
        return {"theil_index": 0.0, **{f"theil_rate_{g}": r for g, r in rates.items()}}
    theil = 0.0
    counts = sensitive.dropna().value_counts()
    total = float(counts.sum())
    for group, rate in rates.items():
        if rate <= 0 or not np.isfinite(rate):
            continue
        group_share = float(counts.get(group, 0) / total) if total else 0.0
        theil += group_share * (rate / overall) * np.log(rate / overall)
    return {
        "theil_index": max(0.0, float(theil)),
        **{f"theil_rate_{g}": r for g, r in rates.items()},
    }


def intersectional_spd(
    y_pred: pd.Series,
    sensitive_dict: dict[str, pd.Series],
    min_group_size: int = 20
) -> dict[str, float]:
    """SPD computed on the Cartesian product of the provided sensitive attrs.

    Treats each unique intersection (e.g. SEX=1 & RAC1P=2 & HISP=2) as a single
    group and reports the range of positive-prediction rates across all
    intersections. This is a thesis-relevant extension of SPD that surfaces
    bias affecting small sub-populations.
    """
    if not sensitive_dict:
        return {"intersect_spd_range": 0.0}
    combined = pd.concat(sensitive_dict, axis=1)
    combined.columns = [str(c) for c in combined.columns]
    keys = combined.astype(str).agg("|".join, axis=1)
    rates: dict[str, float] = {}
    for key in keys.dropna().unique():
        mask = keys == key
        if int(mask.sum()) < min_group_size:
            continue
        rates[str(key)] = _safe_rate(y_pred[mask])
    if not rates:
        return {"intersect_spd_range": 0.0}
    return {
        "intersect_spd_range": float(max(rates.values()) - min(rates.values())),
        "intersect_n_groups": float(len(rates)),
        "intersect_min_group_size": float(min_group_size),
    }


def full_fairness_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive: pd.Series | dict[str, pd.Series],
) -> dict[str, float]:
    """Compute the full set of fairness metrics for a single sensitive attr or
    a dict of {attr_name: series}. The output is a flat dict suitable for
    storing in a row of the fairness summary CSV.
    """
    out: dict[str, float] = {}
    if isinstance(sensitive, pd.Series):
        sensitive = {"sensitive": sensitive}
    for attr, series in sensitive.items():
        prefix = f"{attr}__"
        spd = statistical_parity_difference(y_pred, series)
        eod = equal_opportunity_difference(y_true, y_pred, series)
        di = disparate_impact(y_pred, series)
        ti = theil_index(y_pred, series)
        for k, v in spd.items():
            out[f"{prefix}spd_{k}"] = v
        for k, v in eod.items():
            out[f"{prefix}eod_{k}"] = v
        for k, v in di.items():
            out[f"{prefix}di_{k}"] = v
        for k, v in ti.items():
            out[f"{prefix}theil_{k}"] = v
    intersect = intersectional_spd(
        y_pred,
        {k: v for k, v in (sensitive.items() if isinstance(sensitive, dict) else [])},
    )
    for k, v in intersect.items():
        out[f"intersect__{k}"] = v
    return out


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, y_prob: np.ndarray | None = None) -> ModelMetrics:
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else 0.0
    return ModelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=float(auc),
    )
