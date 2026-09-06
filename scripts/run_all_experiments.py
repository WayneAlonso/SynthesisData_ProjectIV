"""兼容老入口：转发到 main.py（4 周主入口）。

老用法：`python scripts/run_all_experiments.py` —— 当时只跑 ACS 三数据集。
现在：`python main.py` 才是新一周的入口；本脚本保留作为"老习惯兼容层"，
默认行为 = 跑 4 周全部任务（在所有数据集上）。

CLI 选项透传：
    python scripts/run_all_experiments.py --week 1
    python scripts/run_all_experiments.py --datasets sbo
    python scripts/run_all_experiments.py --skip-optuna
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 把 main.py 当模块导入再 runpy 调用，等价于 `python main.py <args>`
import runpy
sys.argv = [sys.argv[0]] + sys.argv[1:]
runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
