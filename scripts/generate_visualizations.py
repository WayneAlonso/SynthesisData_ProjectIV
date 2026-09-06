"""Generate visualizations from W1-W4 outputs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fairness_lab.paths import ARTIFACTS_DIR
from fairness_lab.utils.io import ensure_dir
from fairness_lab.utils.reporting import save_bar_chart, save_heatmap, save_line_chart, save_pareto_front

sns.set_theme(style="whitegrid", context="talk")
PALETTE = ["#1f4e79", "#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f79646"]


def plot_w1_fairness(out_dir: Path) -> None:
    """W1: fairness heatmaps across sensitive attributes."""
    df = pd.read_csv(out_dir / "w1" / "fairness_summary.csv")
    if df.empty:
        print("[viz] W1 fairness_summary empty, skip")
        return

    # SPD across sensitive attributes
    sens_attrs = ["SEX", "RAC1P", "HISP"]
    spd_cols = {a: f"{a}__spd_spd_range" for a in sens_attrs}
    eod_cols = {a: f"{a}__eod_eod_tpr_range" for a in sens_attrs}
    di_cols = {a: f"{a}__di_disparate_impact" for a in sens_attrs}

    spd_data = pd.DataFrame({a: df[c].values for a, c in spd_cols.items() if c in df.columns}, index=df["dataset"])
    eod_data = pd.DataFrame({a: df[c].values for a, c in eod_cols.items() if c in df.columns}, index=df["dataset"])
    di_data = pd.DataFrame({a: df[c].values for a, c in di_cols.items() if c in df.columns}, index=df["dataset"])

    if not spd_data.empty:
        save_heatmap(spd_data, "SPD Range by Sensitive Attribute", out_dir / "w1" / "spd_heatmap.png")
        print("[viz] W1 spd_heatmap.png")
    if not eod_data.empty:
        save_heatmap(eod_data, "EOD TPR Range by Sensitive Attribute", out_dir / "w1" / "eod_heatmap.png")
        print("[viz] W1 eod_heatmap.png")
    if not di_data.empty:
        save_heatmap(di_data, "Disparate Impact by Sensitive Attribute\n(1.0 = parity)", out_dir / "w1" / "di_heatmap.png")
        print("[viz] W1 di_heatmap.png")

    # Per-group positive rates for diagnostic
    rows = []
    for _, r in df.iterrows():
        for a in sens_attrs:
            for c in df.columns:
                if c.startswith(f"{a}__spd_spd_rate_"):
                    grp = c.replace(f"{a}__spd_spd_rate_", "")
                    rows.append({"dataset": r["dataset"], "attribute": a, "group": grp,
                                 "positive_rate": r[c]})
    if rows:
        gdf = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=gdf, x="group", y="positive_rate", hue="attribute", palette=PALETTE[:3], ax=ax)
        ax.set_title("Predicted Positive Rate by Group\n(baseline LGBM)")
        ax.set_ylabel("Predicted positive rate")
        plt.tight_layout()
        plt.savefig(out_dir / "w1" / "group_positive_rates.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[viz] W1 group_positive_rates.png")


def plot_w2_accuracy(out_dir: Path) -> None:
    """W2: model comparison + optuna history."""
    df = pd.read_csv(out_dir / "w2" / "accuracy_summary.csv")
    if df.empty:
        print("[viz] W2 accuracy_summary empty, skip")
        return

    # F1 + accuracy per dataset
    fig, ax = plt.subplots(figsize=(9, 5))
    melted = df[["dataset", "accuracy", "f1", "auc"]].melt(id_vars="dataset", var_name="metric", value_name="value")
    sns.barplot(data=melted, x="dataset", y="value", hue="metric", palette=PALETTE[:3], ax=ax)
    ax.set_title("W2 - Accuracy / F1 / AUC by Dataset")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "w2" / "w2_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[viz] W2 w2_metrics.png")


def plot_w3_fairness(out_dir: Path) -> None:
    """W3: Pareto frontier + method comparison."""
    df = pd.read_csv(out_dir / "w3" / "fairness_vs_accuracy.csv")
    if df.empty:
        print("[viz] W3 fairness_vs_accuracy empty, skip")
        return

    # Pareto front per dataset: pick a SPD column
    sens_col = None
    for c in df.columns:
        if c.endswith("__spd_spd_range"):
            sens_col = c
            break

    if sens_col and "accuracy" in df.columns:
        pareto_df = df[["dataset", "method", "lambda", "accuracy", sens_col]].dropna(subset=["accuracy"])
        for ds, sub in pareto_df.groupby("dataset"):
            sub2 = sub.rename(columns={sens_col: "spd"}).copy()
            sub2["spd_abs"] = sub2["spd"].abs()
            save_pareto_front(sub2, x="spd_abs", y="accuracy", label_col="method",
                              title=f"Pareto: |{sens_col}| vs Accuracy ({ds})",
                              path=out_dir / "w3" / f"pareto_{ds}.png")
            print(f"[viz] W3 pareto_{ds}.png")

    # Method comparison bar
    valid = df[df["method"].notna()].copy()
    if not valid.empty and "accuracy" in valid.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=valid, x="method", y="accuracy", hue="dataset", palette=PALETTE, ax=ax)
        ax.set_title("W3 - Accuracy by Fairness Method")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "w3" / "method_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[viz] W3 method_accuracy.png")


def plot_w4_synth(out_dir: Path) -> None:
    """W4: TSTR vs TRTR, privacy-utility tradeoff."""
    df = pd.read_csv(out_dir / "w4" / "synth_evaluation.csv")
    if df.empty:
        print("[viz] W4 synth_evaluation empty, skip")
        return

    # 1. TSTR accuracy by synthesizer (privacy=0)
    no_dp = df[df["privacy_level"] == 0.0].copy()
    if not no_dp.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=no_dp, x="synthesizer", y="tstr_accuracy", hue="dataset",
                    palette=PALETTE[:2], ax=ax)
        ax.axhline(no_dp["trtr_accuracy"].mean(), color="red", linestyle="--",
                   label=f"TRTR mean = {no_dp['trtr_accuracy'].mean():.3f}")
        ax.set_title("W4 - TSTR Accuracy by Synthesizer (no DP)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "w4" / "tstr_by_synth.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[viz] W4 tstr_by_synth.png")

    # 2. Privacy-utility tradeoff
    if "tstr_accuracy" in df.columns and "privacy_level" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=df, x="privacy_level", y="tstr_accuracy", hue="synthesizer",
                     marker="o", palette=PALETTE, ax=ax)
        ax.set_title("W4 - Privacy (DP noise) vs Utility (TSTR Accuracy)")
        ax.set_xlabel("Privacy noise level σ")
        ax.set_ylabel("TSTR accuracy")
        ax.legend(frameon=False, title="Synthesizer")
        plt.tight_layout()
        plt.savefig(out_dir / "w4" / "privacy_utility.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[viz] W4 privacy_utility.png")

    # 3. TSTR vs TRTR scatter
    if "tstr_auc" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x="trtr_accuracy", y="tstr_accuracy",
                        hue="synthesizer", style="privacy_level", palette=PALETTE, s=120, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("TRTR accuracy (real → real)")
        ax.set_ylabel("TSTR accuracy (synth → real)")
        ax.set_title("W4 - TSTR vs TRTR Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "w4" / "tstr_vs_trtr.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[viz] W4 tstr_vs_trtr.png")


def main() -> None:
    out_dir = ARTIFACTS_DIR
    print(f"[viz] output dir = {out_dir}")
    plot_w1_fairness(out_dir)
    plot_w2_accuracy(out_dir)
    plot_w3_fairness(out_dir)
    plot_w4_synth(out_dir)
    print("[viz] done")


if __name__ == "__main__":
    main()