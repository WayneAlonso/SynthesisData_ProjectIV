# 整理版验证记录

验证日期：2026-09-02。环境：Windows、Python 3.10.20、CPU。直接依赖版本记录在根目录的 `requirements-tested.txt`；依赖检查通过。

## 已执行

| 检查 | 结果 |
| --- | --- |
| 原始数据 SHA-256、字节数、CSV 行列数 | 24 个数据目录文件通过，含 8 份 CSV |
| 默认配置数据路径 | 5 个数据集均存在实际数据 |
| 数据恢复 | `ma2018.csv` 从原压缩包恢复，与原 LFS 指针的 SHA-256 一致 |
| 重复数据 | 移除的 3 份 ACS 根目录副本与地区文件内容一致 |
| 重复结果 | W4 checkpoint 与最终 CSV 内容一致，只保留最终文件 |
| Python 语法 | 47 个 Python 文件通过解析 |
| 命令行入口 | 18 个入口的 `--help` 正常退出 |
| 现有回归测试 | `5 passed` |
| 实际运行 W1 | Massachusetts 2019，读取真实文件并生成基线结果 |
| 实际运行 W4 | Massachusetts 2019，Gaussian Copula，噪声强度 0.0，生成 6,104 条合成记录并完成 TSTR 评估 |
| 交付文件检查 | 无论文文档、PDF、LaTeX、同步副本、缓存、可执行工具、ZIP 或未恢复的 LFS 指针 |
| 普通 Git 文件大小 | 所有文件小于 100 MiB |

回归测试覆盖标签源字段排除、DI、Theil、留出数据分离和噪声处理对标签及类别字段的保护。论文结构配置的两个文档测试随论文专用配置一同排除；实验代码测试全部保留。

## 实际运行记录

W1 验证命令为根 README 的 baseline 示例，输出目录通过 `--out-root` 指向交付目录之外的临时验证目录。结果 `source=file`，Accuracy 约 0.8735、F1 约 0.8414、AUC 约 0.9481。

W4 使用根 README 的 Gaussian Copula 示例，并将 `--out` 指向交付目录之外的临时目录。合成器 backend 为 `sdv`，TSTR Accuracy 约 0.7936、F1 约 0.7454、AUC 约 0.8555。此处数值只用于确认流程执行，不作为新一轮完整模型比较结果。

**W4 质量计算的已知限制：**当前混合字段类型触发 SDMetrics 相关性指标错误 `data type <class 'numpy.object_'> not inexact`，代码按既有逻辑记录错误并使用 `proxy_fallback` 质量指标。合成和下游预测评估已完成，但本次不能报告为所有质量指标都由 SDMetrics 成功计算。

## 验证范围

- 未重新运行所有数据集、全部合成器、完整 Optuna 搜索或所有随机种子的长期实验。
- 本次使用已有依赖环境验证，未在全新虚拟环境中重新安装全部依赖，也未验证 macOS / Linux 或 CUDA。
- `--help` 通过表示入口能加载并解析参数，不等于所有旧版流程均经过端到端验证。
- 既有 `results/` 的数值原样保留，不与本次临时验证结果混合。
- 实验性直方图合成器未经过正式差分隐私审计；本交付不新增其数学保证。

重新验证可运行 `python scripts/verify_data.py`、`python -m pytest tests -q`，再执行根 README 中的 baseline 和 Gaussian Copula 示例。
