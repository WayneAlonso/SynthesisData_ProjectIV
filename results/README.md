# 已有实验结果

本目录保存整理前项目中的 CSV / JSON 结果快照，数值未重新计算。

| 目录 | 主要结果 |
| --- | --- |
| `w1/` | 基线公平性、外部留出审计、漂移和痛点表 |
| `w2/` | 准确率汇总、Optuna 状态及逐数据集结果 |
| `w3/` | 公平性与效用比较、Pareto 前沿、lambda 配置 |
| `w4/` | 合成器效用、公平性与质量评估表 |
| `w3_repeated_acs_ma_2seed/` | Massachusetts 2019、seed 42 / 43 的逐次结果、汇总及群体诊断 |

快照可能来自不同运行批次与参数，不应当作本次验证产生的完整新实验。错误字段、缺失值和实际 backend 是结果的一部分。两种子结果不能替代充分重复实验。

W4 的中间 checkpoint 与最终 CSV 重复，因此只保留最终 CSV。论文、历史 Markdown 报告、临时 smoke 结果与聊天记录未纳入此目录。

新实验写入 `artifacts/`；`scripts/generate_final_report.py` 和可视化脚本读取 `artifacts/`，不会自动读取这些历史快照。
