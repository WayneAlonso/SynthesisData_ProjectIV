from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io import ensure_dir


THESIS_PALETTE = ["#1f4e79", "#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f79646"]


def _prepare_canvas(figsize: tuple[float, float] = (10, 5)) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=figsize)


def _finalize_plot(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    path: Path,
    hue: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    _prepare_canvas(figsize=figsize)
    hue_key = hue or x
    series_count = max(1, df[hue_key].nunique(dropna=False))
    sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue_key,
        dodge=hue is not None,
        palette=THESIS_PALETTE[:series_count],
    )
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    if hue is None:
        legend = plt.gca().get_legend()
        if legend is not None:
            legend.remove()
    else:
        plt.legend(frameon=False)
    _finalize_plot(path)


def save_line_chart(
    df: pd.DataFrame,
    x: str,
    y_columns: list[str],
    title: str,
    path: Path,
    figsize: tuple[float, float] = (10, 5),
) -> None:
    _prepare_canvas(figsize=figsize)
    plot_df = df[[x] + y_columns].melt(id_vars=x, value_vars=y_columns, var_name="metric", value_name="value")
    series_count = max(1, len(y_columns))
    sns.lineplot(
        data=plot_df,
        x=x,
        y="value",
        hue="metric",
        style="metric",
        markers=True,
        dashes=False,
        palette=THESIS_PALETTE[:series_count],
    )
    plt.title(title)
    plt.legend(frameon=False)
    _finalize_plot(path)


def save_heatmap(df: pd.DataFrame, title: str, path: Path, figsize: tuple[float, float] = (10, 6)) -> None:
    _prepare_canvas(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"shrink": 0.85})
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    _finalize_plot(path)


def save_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str,
    title: str,
    path: Path,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    _prepare_canvas(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, s=180, color=THESIS_PALETTE[0])
    for _, row in df.iterrows():
        plt.text(row[x], row[y], str(row[label_col]), fontsize=10, ha="left", va="bottom")
    plt.title(title)
    _finalize_plot(path)


def save_pareto_front(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str,
    title: str,
    path: Path,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    """Scatter plot highlighting Pareto-optimal points."""
    _prepare_canvas(figsize=figsize)
    pareto_x = df[x].values
    pareto_y = df[y].values
    idx = np.argsort(pareto_x)
    sx, sy = pareto_x[idx], pareto_y[idx]
    max_y = -np.inf
    pareto_mask = np.zeros(len(sx), dtype=bool)
    for i in range(len(sx) - 1, -1, -1):
        if sy[i] >= max_y:
            max_y = sy[i]
            pareto_mask[i] = True
    plt.plot(sx[pareto_mask], sy[pareto_mask], "o-", color=THESIS_PALETTE[1], linewidth=2, markersize=8, label="Pareto front")
    non_pareto = ~pareto_mask
    if non_pareto.any():
        plt.scatter(sx[non_pareto], sy[non_pareto], color=THESIS_PALETTE[0], s=80, alpha=0.5, label="Dominated")
    for _, row in df.iterrows():
        plt.text(row[x], row[y], str(row[label_col]), fontsize=9, ha="left", va="bottom")
    plt.title(title)
    plt.legend(frameon=False)
    _finalize_plot(path)


def dataframe_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(col) for col in df.columns]
        rows = df.astype(str).values.tolist()
        table = ["| " + " | ".join(headers) + " |"]
        table.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            table.append("| " + " | ".join(row) + " |")
        return "\n".join(table)
