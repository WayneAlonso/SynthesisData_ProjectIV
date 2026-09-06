from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fairness_lab.experiments.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fairness benchmark pipeline.")
    parser.add_argument("--dataset", default="acs_ma_2019", help="Dataset name from configs/experiment.json")
    parser.add_argument("--source", default="auto", choices=["auto", "file", "demo"], help="Data source preference")
    args = parser.parse_args()

    result = run_full_pipeline(dataset_name=args.dataset, source=args.source)
    print(f"Completed pipeline for {result['dataset']} with source={result['source']}.")
    print(f"Run id: {result['run_id']}")


if __name__ == "__main__":
    main()
