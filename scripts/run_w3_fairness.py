"""W3 提公平性：fairlearn 约束训练 + 后处理 + OOD 审计 + 帕累托前沿。

执行：
    python scripts/run_w3_fairness.py [--datasets sbo] [--lambdas 0.1 0.5 1.0 2.0 5.0]
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

from w3_fairness import W3Fairness  # noqa: E402


DEFAULT_DATASETS = ["sbo", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"]
DEFAULT_LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0]


def main() -> int:
    parser = argparse.ArgumentParser(description="W3 提公平性")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS,
                        help="DemographicParity 约束的 λ 列表（控制公平-效用权衡）")
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "w3"))
    args = parser.parse_args()

    print("=" * 60)
    print(f"W3 提公平性  (λ={args.lambdas})")
    print("=" * 60)

    runner = W3Fairness(out_dir=Path(args.out), lambdas=args.lambdas)
    runner.run(datasets=args.datasets)

    print()
    print(f"✅ W3 全部完成。报告: {Path(args.out) / 'pareto_frontier.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())