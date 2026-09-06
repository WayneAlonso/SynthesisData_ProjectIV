# 运行脚本

推荐从根目录的 `research_main.py` 开始。所有脚本均在项目根目录运行。

| 脚本 | 用途 |
| --- | --- |
| `install_env.py` | 根据根 `requirements.txt` 安装依赖 |
| `check_env.py` | 检查实验依赖 |
| `verify_data.py` | 校验数据文件、行列数和配置路径 |
| `run_w1_diagnosis.py` | 基线与外部审计 |
| `run_w2_accuracy.py` | 调参与模型比较 |
| `run_w3_fairness.py` | 公平性优化 |
| `run_w4_synth.py` | 按 `--synthesizers` 选择合成器 |
| `generate_final_report.py` | 汇总 `artifacts/` 中的 W1–W4 结果 |
| `generate_visualizations.py` | 从 `artifacts/` 结果绘图 |
| `run_all.py` / `run_all_experiments.py` | 保留的批处理及兼容入口 |
| `run_pipeline.py` / `run_sensitivity.py` | 保留的旧版通用实验入口 |

参数详情使用 `python scripts/对应脚本.py --help` 查看；安装、环境检查、数据校验和报告/绘图脚本直接运行即可。
