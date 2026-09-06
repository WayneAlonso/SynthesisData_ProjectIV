"""Synthetic Data Fairness 一键入口：按四周计划顺序跑 W1 → W2 → W3 → W4。

用法：
    python main.py                       # 默认全跑（5 数据集 + SBO withheld）
    python main.py --week 1              # 只跑 W1
    python main.py --week 2 3            # 跑 W2+W3
    python main.py --datasets sbo        # 只在 sbo 上跑
    python main.py --skip-optuna         # W2 跳过 Optuna（省时间）
"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fairness_lab.paths import ARTIFACTS_DIR  # noqa: E402
from fairness_lab.utils.io import ensure_dir  # noqa: E402
from fairness_lab.settings import Settings  # noqa: E402


DEFAULT_DATASETS = ["sbo", "sbo_withheld", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"]


def _banner(week: str, title: str) -> None:
    print()
    print("=" * 70)
    print(f" {week}：{title}")
    print("=" * 70)


def _run_w1(datasets: list[str], out_dir: Path) -> None:
    _banner("第 1 周 (W1)", "诊断与基线 - 5 项公平性指标 + OOD 审计 + 痛点表")
    from w1_diagnosis import W1Diagnosis
    W1Diagnosis(out_dir=out_dir).run(datasets=datasets)


def _run_w2(datasets: list[str], out_dir: Path, skip_optuna: bool) -> None:
    _banner("第 2 周 (W2)", "提准确率 - Optuna 调参 + 3 折 CV + 模型对比")
    from w2_accuracy import W2Accuracy
    settings = Settings()
    datasets = [d for d in datasets if settings.dataset(d).role != "holdout_audit"]
    W2Accuracy(out_dir=out_dir, n_trials=settings.optuna_n_trials).run(
        datasets=datasets, skip_optuna=skip_optuna,
    )


def _run_w3(datasets: list[str], out_dir: Path) -> None:
    _banner("第 3 周 (W3)", "提公平性 - fairlearn DP/EO + ThresholdOptimizer + 交叉组 + 帕累托")
    from w3_fairness import W3Fairness
    settings = Settings()
    datasets = [d for d in datasets if settings.dataset(d).role != "holdout_audit"]
    W3Fairness(out_dir=out_dir, lambdas=settings.fairlearn_lambdas).run(datasets=datasets)


def _run_w4(datasets: list[str], out_dir: Path) -> None:
    _banner("第 4 周 (W4)", "提合成数据可用性 - 4 合成器 + sdmetrics + TSTR + 噪声敏感性分析")
    from w4_synth import W4Synth
    settings = Settings()
    # 排除 holdout 数据集，holdout 只用于 W1 OOD 审计
    train_ds = [d for d in datasets if Settings().dataset(d).role != "holdout_audit"]
    W4Synth(
        out_dir=out_dir,
        n_synth=settings.n_synth,
        privacy_levels=settings.privacy_levels,
    ).run(datasets=train_ds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic Data Fairness 一键跑四周计划")
    parser.add_argument("--week", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="要跑的周次（默认 1 2 3 4）")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help=f"数据集列表（默认：{' '.join(DEFAULT_DATASETS)}）")
    parser.add_argument("--out-root", default=str(ARTIFACTS_DIR),
                        help="artifacts 根目录")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="W2 跳过 Optuna 调参（默认 LGBM 默认参数）")
    args = parser.parse_args()

    out_root = ensure_dir(Path(args.out_root))

    print(f"Synthetic Data Fairness - 四周计划：数据集 = {args.datasets}")
    print(f"输出根目录: {out_root}")
    print(f"周次:       {args.week}")
    if args.skip_optuna:
        print("W2 调参:    跳过（使用默认参数）")
    print()

    t0 = time.time()
    if 1 in args.week:
        _run_w1(args.datasets, out_root / "w1")
    if 2 in args.week:
        _run_w2(args.datasets, out_root / "w2", skip_optuna=args.skip_optuna)
    if 3 in args.week:
        _run_w3(args.datasets, out_root / "w3")
    if 4 in args.week:
        _run_w4(args.datasets, out_root / "w4")

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f" ✅ 全部完成，耗时 {elapsed/60:.1f} 分钟")
    print("=" * 70)
    print()
    print("各周产出：")
    for w in args.week:
        d = out_root / f"w{w}"
        if d.exists():
            print(f"  W{w} → {d}")
    print()
    print("下一步：python scripts/generate_final_report.py  汇总 4 周产出 → FINAL_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())