"""W2 提准确率：Optuna 调参 + TargetEncoder + 3 折 CV。

执行：
    python scripts/run_w2_accuracy.py [--datasets sbo acs_national_2019 ...] [--n-trials 100]

产出：
    artifacts/w2/
        optuna_best_params.json      每个数据集的最优 LGBM 超参
        optuna_study_<ds>.db         Optuna study（可后续可视化）
        accuracy_summary.csv               3 折 CV：accuracy/f1/auc ± std
        model_comparison.csv         LGBM vs XGBoost vs CatBoost 对比
        accuracy_improvement.png     W1 vs W2 准召率提升柱状图
"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from w2_accuracy import W2Accuracy  # noqa: E402


DEFAULT_DATASETS = ["sbo", "acs_national_2019"]


def main() -> int:
    parser = argparse.ArgumentParser(description="W2 提准确率")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna 试验次数（默认 50）")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "w2"))
    parser.add_argument("--skip-optuna", action="store_true", help="跳过调参，用默认参数")
    args = parser.parse_args()

    print("=" * 60)
    print(f"W2 提准确率  (Optuna trials={args.n_trials})")
    print("=" * 60)

    runner = W2Accuracy(out_dir=Path(args.out), n_trials=args.n_trials)
    runner.run(datasets=args.datasets, skip_optuna=args.skip_optuna)

    print()
    print(f"✅ W2 全部完成。报告: {Path(args.out) / 'accuracy_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())