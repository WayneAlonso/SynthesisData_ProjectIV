"""Generate a report from the current Synthetic Data Fairness artifact names."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairness_lab.paths import ARTIFACTS_DIR  # noqa: E402
from fairness_lab.utils.io import ensure_dir  # noqa: E402


def read_text(path: Path, default: str = "（未生成）") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default


def table(path: Path) -> str:
    if not path.exists():
        return "（未生成）"
    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        return "（空表）"
    try:
        return df.round(4).to_markdown(index=False)
    except ImportError:
        return "```\n" + df.round(4).to_string(index=False) + "\n```"


def json_block(path: Path) -> str:
    if not path.exists():
        return "{}"
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)


def main() -> int:
    art = ARTIFACTS_DIR
    ensure_dir(art)
    sections: list[str] = [
        "# Synthetic Data Fairness 实验结果总报告",
        "",
        "> 本报告读取当前版本实际生成的 W1–W4 文件，不再使用旧版文件名。",
        "",
        "## 1. W1 诊断与基线",
        "",
        read_text(art / "w1" / "W1_DIAGNOSIS.md"),
        "",
        "### W1 痛点表",
        "",
        table(art / "w1" / "pain_points.csv"),
        "",
        "## 2. W2 准确率与调参",
        "",
        "### 准确率汇总",
        "",
        table(art / "w2" / "accuracy_summary.csv"),
        "",
        "### Optuna 状态",
        "",
        "```json",
        json_block(art / "w2" / "optuna_results.json"),
        "```",
        "",
        "## 3. W3 公平性优化",
        "",
        table(art / "w3" / "fairness_vs_accuracy.csv"),
        "",
        "### Pareto 前沿",
        "",
        table(art / "w3" / "pareto_frontier.csv"),
        "",
        "## 4. W4 合成数据可用性",
        "",
        read_text(art / "w4" / "W4_SYNTH.md"),
        "",
        "### 合成数据评估明细",
        "",
        table(art / "w4" / "synth_evaluation.csv"),
        "",
        "## 5. 可复现性说明",
        "",
        "- `quality_backend=sdmetrics` 表示使用 SDMetrics；`proxy_fallback` 表示使用了明确标注的本地回退指标。",
        "- W4 的 privacy_level 是数值加噪敏感性分析，不等同于严格差分隐私。",
        "- 如果某阶段没有结果，报告会显示“未生成”，不会再把缺失结果伪装成已完成。",
        "",
    ]
    output = art / "FINAL_REPORT.md"
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"报告已生成: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())