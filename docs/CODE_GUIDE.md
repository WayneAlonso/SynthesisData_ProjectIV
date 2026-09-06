# 代码导航

## 推荐入口

`research_main.py` 将实验分成 `baseline`、`model_selection`、`fairness_mitigation`、`synthesis`，并按配置的 `holdout_audit` 角色分离训练数据与外部留出数据。`main.py` 提供对应的 `--week 1 2 3 4` 参数。

| 文件或目录 | 作用 |
| --- | --- |
| `src/fairness_lab/paths.py` | 根据项目位置推导数据、配置和输出目录 |
| `src/fairness_lab/settings.py` | 读取实验配置及数据集定义 |
| `src/fairness_lab/data/loader.py` | CSV 加载、标签转换、旧版演示数据 |
| `src/fairness_lab/data/preprocessing.py` | 特征筛选、训练测试划分、缺失值和编码处理 |
| `src/fairness_lab/data/profiling.py` | 数据结构、缺失值和分组描述 |
| `src/fairness_lab/fairness/metrics.py` | 效用指标及群体、交叉群体公平性指标 |
| `src/fairness_lab/models/baseline.py` | 预测模型及比较方法 |
| `src/fairness_lab/models/generative.py` | SDV 合成器适配器及显式回退实现 |
| `src/fairness_lab/utils/` | 文件写入、表格与图形输出 |
| `src/fairness_lab/experiments/` | 保留的旧版通用 pipeline 与敏感性分析 |
| `w1_diagnosis.py` | 基线及 target→withheld 审计 |
| `w2_accuracy.py` | Optuna、交叉验证和模型比较 |
| `w3_fairness.py` | Fairlearn 约束与后处理 |
| `w3_repeated.py` | 多种子汇总、bootstrap 区间与群体诊断 |
| `w3_sparse.py` | 原始、合并、重采样群体处理比较 |
| `w4_synth.py` | 合成、质量、TSTR、公平性及噪声强度比较 |
| `w4_repeated.py` | 多种子合成实验及均值、标准差 |
| `w4_formal_dp.py` / `formal_dp_synthesizer.py` | 实验性独立边缘直方图合成流程 |
| `scripts/verify_data.py` | 校验数据内容和配置路径 |
| `scripts/run_w*.py` | 各阶段独立命令行入口 |
| `scripts/generate_final_report.py` | 汇总 `artifacts/` 中已有结果 |
| `scripts/generate_visualizations.py` | 从 `artifacts/` 表格生成图形 |

## 兼容脚本

`scripts/run_all_experiments.py` 转发到 `main.py`。`scripts/run_all.py` 是保留的旧版批处理入口。`run_pipeline.py` 和 `run_sensitivity.py` 使用旧版通用流程，其分析步骤与 W1–W4 不完全相同；新实验使用根 README 中的推荐命令。

各底层类和扩展脚本仍可接收显式数据集参数。外部留出隔离由推荐入口统一控制，扩展训练实验应只传入训练数据集。不要将 `sbo_withheld` 显式传入调参、公平性训练或合成训练。

## 本次整理涉及的代码调整

- 清除同步副本、旧项目版本标题和指向已排除论文脚本的后续运行提示。
- 为 `main.py` 的 W2 / W3 入口补上按数据集角色过滤留出数据。
- 修复 W4 包装脚本把整数样本量传给“合成器名称列表”参数的问题，改为 `--synthesizers`；样本量沿用核心实现，等于真实训练集行数。
- 修复通用 pipeline 中不存在的 `settings.synthesis_models` 属性引用，以及旧脚本中不存在的默认数据集 `acs_income`。
- 统一环境安装脚本与 `requirements.txt`，补充直接使用的 SciPy 和 openpyxl。
- 原项目中的论文结构配置、配套文档测试以及文档和参考文献生成脚本不属于本交付版。保留、排除和数据恢复清单见 `export_manifest.json`。

本次以可搬移的代码与数据交付为范围，没有重新设计统计指标、训练算法或正式隐私机制。
