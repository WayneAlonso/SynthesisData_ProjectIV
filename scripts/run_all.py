"""一键跑通 4 周计划全部实验。

执行：
    python scripts/run_all.py [--skip-optuna] [--datasets sbo sbo_withheld acs_national_2019]

跑完后会生成 `artifacts/FINAL_REPORT.md`，含三线达标对照表 + 论文摘要段落。
"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="一键跑通 4 周计划")
    parser.add_argument("--datasets", nargs="+", default=["sbo", "sbo_withheld", "acs_national_2019", "acs_ma_2019", "acs_tx_2019"])
    parser.add_argument("--skip-optuna", action="store_true", help="跳过 W2 Optuna 调参")
    parser.add_argument("--skip-w3", action="store_true", help="跳过 W3 fairlearn 训练")
    parser.add_argument("--skip-w4", action="store_true", help="跳过 W4 合成数据实验")
    args = parser.parse_args()

    print("=" * 70)
    print("Synthetic Data Fairness · 一键跑通 4 周计划")
    print(f"数据集: {args.datasets}")
    print("=" * 70)

    # ---------- W1 诊断 ----------
    print("\n" + "=" * 70)
    print("[W1] 诊断与基线")
    print("=" * 70)
    from w1_diagnosis import W1Diagnosis
    W1Diagnosis(out_dir=ROOT / "artifacts" / "w1").run(datasets=args.datasets)

    # ---------- W1.4 文档 ----------
    print("\n" + "=" * 70)
    print("[W1.4] loan_approved 派生逻辑文档")
    print("=" * 70)
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_target_doc  # type: ignore
    run_target_doc.main() if hasattr(run_target_doc, "main") else None
    # 直接调用函数
    from render_target_doc import main as render_target_doc_main
    render_target_doc_main()

    # ---------- W2 准确率 ----------
    print("\n" + "=" * 70)
    print("[W2] 提准确率（Optuna + 3 折 CV）")
    print("=" * 70)
    from w2_accuracy import W2Accuracy
    W2Accuracy(out_dir=ROOT / "artifacts" / "w2", n_trials=30).run(
        datasets=args.datasets, skip_optuna=args.skip_optuna,
    )

    # ---------- W3 公平性 ----------
    if not args.skip_w3:
        print("\n" + "=" * 70)
        print("[W3] 提公平性（fairlearn + 帕累托 + 交叉组）")
        print("=" * 70)
        from w3_fairness import W3Fairness
        W3Fairness(
            out_dir=ROOT / "artifacts" / "w3",
            lambdas=[0.1, 0.5, 1.0, 2.0],
        ).run(datasets=[d for d in args.datasets if d != "sbo_withheld"])

    # ---------- W4 合成数据 ----------
    if not args.skip_w4:
        print("\n" + "=" * 70)
        print("[W4] 提合成数据可用性（sdmetrics + TSTR + 隐私档）")
        print("=" * 70)
        from w4_synth import W4Synth
        W4Synth(
            out_dir=ROOT / "artifacts" / "w4",
            n_synth=["ctgan", "tvae", "gaussian_copula", "copula_gan"],
            privacy_levels=[0.0, 0.1, 0.3],
        ).run(datasets=[d for d in args.datasets if d != "sbo_withheld"])

    # ---------- Final Report ----------
    print("\n" + "=" * 70)
    print("[FINAL] 汇总报告")
    print("=" * 70)
    _render_final_report()

    print("\n" + "=" * 70)
    print("🎉 全部完成。")
    print(f"  最终报告: {ROOT / 'artifacts' / 'FINAL_REPORT.md'}")
    print("=" * 70)
    return 0


def _render_final_report():
    """Use the single report generator that matches the current artifact schema."""
    from generate_final_report import main as generate_report
    return generate_report()

if __name__ == "__main__":
    sys.exit(main())