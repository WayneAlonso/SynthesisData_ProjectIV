"""Run selected synthesizers with optional numeric noise sensitivity levels."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fairness_lab.settings import Settings
from w4_synth import W4Synth

def main():
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["sbo", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"])
    parser.add_argument("--synthesizers", nargs="+", choices=settings.n_synth, default=settings.n_synth)
    parser.add_argument("--privacy", nargs="+", type=float, default=settings.privacy_levels,
                        help="Numeric noise strengths; these are not differential privacy budgets.")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "w4")
    args = parser.parse_args()
    datasets = [d for d in args.datasets if settings.dataset(d).role != "holdout_audit"]
    if not datasets:
        parser.error("At least one training dataset is required.")
    W4Synth(out_dir=args.out, n_synth=args.synthesizers,
            privacy_levels=args.privacy, use_gpu=args.use_gpu).run(datasets=datasets)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
