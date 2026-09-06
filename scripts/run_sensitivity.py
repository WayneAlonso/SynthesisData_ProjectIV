from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fairness_lab.experiments.sensitivity import run_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity experiments.")
    parser.add_argument("--dataset", default="acs_ma_2019", help="Dataset name from configs/experiment.json")
    parser.add_argument("--source", default="auto", choices=["auto", "file", "demo"], help="Data source preference")
    args = parser.parse_args()

    result = run_sensitivity(dataset_name=args.dataset, source=args.source)
    print(f"Completed sensitivity experiments for {result['dataset']}.")
    print(f"Run id: {result['run_id']}")


if __name__ == "__main__":
    main()
