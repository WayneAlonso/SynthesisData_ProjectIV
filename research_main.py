"""Canonical experiment entry point with holdout isolation.

This entry preserves the existing W1-W4 implementations while enforcing the
research protocol: holdout datasets may be evaluated by W1's target-to-holdout
audit, but they are never independently trained, tuned, mitigated, or used to
fit a synthesizer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fairness_lab.paths import ARTIFACTS_DIR  # noqa: E402
from fairness_lab.settings import Settings  # noqa: E402
from fairness_lab.utils.io import ensure_dir  # noqa: E402


DEFAULT_DATASETS = [
    "sbo",
    "sbo_withheld",
    "acs_national_2019",
    "acs_ma_2019",
    "acs_tx_2019",
]
STAGE_ORDER = ["baseline", "model_selection", "fairness_mitigation", "synthesis"]


def partition_datasets(dataset_names: list[str], settings: Settings) -> tuple[list[str], list[str]]:
    unknown = [name for name in dataset_names if name not in settings.raw.get("dataset_catalog", {})]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    training = [name for name in dataset_names if settings.dataset(name).role != "holdout_audit"]
    holdouts = [name for name in dataset_names if settings.dataset(name).role == "holdout_audit"]
    return training, holdouts


def _run_baseline(dataset_names: list[str], out_dir: Path) -> None:
    from w1_diagnosis import W1Diagnosis

    W1Diagnosis(out_dir=out_dir).run(datasets=dataset_names)


def _run_model_selection(dataset_names: list[str], out_dir: Path, skip_optuna: bool) -> None:
    from w2_accuracy import W2Accuracy

    settings = Settings()
    W2Accuracy(out_dir=out_dir, n_trials=settings.optuna_n_trials).run(
        datasets=dataset_names,
        skip_optuna=skip_optuna,
    )


def _run_fairness_mitigation(dataset_names: list[str], out_dir: Path) -> None:
    from w3_fairness import W3Fairness

    settings = Settings()
    W3Fairness(out_dir=out_dir, lambdas=settings.fairlearn_lambdas).run(datasets=dataset_names)


def _run_synthesis(dataset_names: list[str], out_dir: Path) -> None:
    from w4_synth import W4Synth

    settings = Settings()
    W4Synth(
        out_dir=out_dir,
        n_synth=settings.n_synth,
        privacy_levels=settings.privacy_levels,
    ).run(datasets=dataset_names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiments with holdout isolation.")
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--out-root", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--skip-optuna", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    training, holdouts = partition_datasets(args.datasets, settings)
    if not training:
        parser.error("At least one training dataset is required.")

    out_root = ensure_dir(args.out_root)
    print(f"Research stages: {args.stages}")
    print(f"Training datasets: {training}")
    print(f"External holdouts: {holdouts or 'none'}")
    if holdouts:
        print("Protocol: holdouts enter baseline external audit only; training stages exclude them.")

    started = time.time()
    if "baseline" in args.stages:
        _run_baseline(training + holdouts, out_root / "w1")
    if "model_selection" in args.stages:
        _run_model_selection(training, out_root / "w2", args.skip_optuna)
    if "fairness_mitigation" in args.stages:
        _run_fairness_mitigation(training, out_root / "w3")
    if "synthesis" in args.stages:
        _run_synthesis(training, out_root / "w4")

    print(f"Completed in {(time.time() - started) / 60:.1f} minutes.")
    print("Next: python scripts/generate_final_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
