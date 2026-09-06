"""W1 诊断与基线：跑通 ACS 三数据集 + SBO + withheld holdout 公平性审计。

执行：
    python scripts/run_w1_diagnosis.py [--datasets sbo sbo_withheld acs_national_2019 ...]

产出：
    artifacts/w1/
        *_baseline_metrics.csv      baseline 模型在每个数据集上的 accuracy/f1/auc
        *_fairness_summary.csv      5 项公平性指标汇总
        *_ood_fairness.csv          (target→withheld) 分布外公平性审计
        *_group_drift.csv           每个子群在 target vs withheld 上的 SPD 漂移
        W1_diagnosis.md             实验诊断报告
        spd_heatmap.png / eod_heatmap.png  公平性热力图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from w1_diagnosis import W1Diagnosis  # noqa: E402


DEFAULT_DATASETS = [
    "sbo",
    "sbo_withheld",
    "acs_national_2019",
    "acs_ma_2019",
    "acs_tx_2019",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="W1 诊断与基线")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"要跑的数据集（默认: {' '.join(DEFAULT_DATASETS)}）",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "w1"),
        help="输出目录",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("W1 诊断与基线")
    print("=" * 60)
    print(f"数据集: {args.datasets}")
    print(f"输出:   {args.out}")
    print()

    runner = W1Diagnosis(out_dir=Path(args.out))
    runner.run(datasets=args.datasets)

    print()
    print(f"W1 completed. Report: {Path(args.out) / 'W1_DIAGNOSIS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())