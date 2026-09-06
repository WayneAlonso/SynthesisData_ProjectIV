"""环境自检脚本：W1.1 配套

检测 4 周计划所需的关键依赖是否就绪。
用法：
    python scripts/check_env.py
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = [
    ("pandas",            "pandas"),
    ("numpy",             "numpy"),
    ("scikit-learn",      "sklearn"),
    ("lightgbm",          "lightgbm"),
    ("xgboost",           "xgboost"),
    ("catboost",          "catboost"),
    ("fairlearn",         "fairlearn"),
    ("sdv",               "sdv"),
    ("sdmetrics",         "sdmetrics"),
    ("optuna",            "optuna"),
    ("matplotlib",        "matplotlib"),
    ("seaborn",           "seaborn"),
    ("openpyxl",          "openpyxl"),
    ("tabulate",          "tabulate"),
]


def main() -> int:
    print("=" * 60)
    print("Synthetic Data Fairness · 环境自检")
    print("=" * 60)

    ok, fail = [], []
    for name, module in CHECKS:
        try:
            m = __import__(module)
            ver = getattr(m, "__version__", "?")
            print(f"  ✅ {name:<14} {ver}")
            ok.append(name)
        except ImportError as e:
            print(f"  ❌ {name:<14} 缺失：{e}")
            fail.append(name)

    print()
    print(f"结果：{len(ok)} 个就绪 / {len(fail)} 个缺失")
    if fail:
        print("\n缺失依赖，请先执行：python scripts/install_env.py")
        return 1
    print("\n🎉 环境就绪，可以跑 4 周计划的全部实验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())