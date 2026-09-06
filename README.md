# Synthetic Data Fairness — 合成数据公平性实验项目

This is a Python-based tabular data experiment project designed to compare the utility, group fairness, and performance of predictive models when used for downstream training on synthetic data. The project includes runnable code, ACS/SBO benchmark data, configuration, regression tests, and existing experimental results tables. 一个基于 Python 的表格数据实验项目，用于比较预测模型的效用、群体公平性，以及合成数据用于下游训练时的表现。项目包含可运行代码、ACS / SBO 基准数据、配置、回归测试和已有实验结果表。

## 功能

| Module | Function | Main Implementation |
| --- | --- | --- |
| W1 | Baseline prediction, group fairness diagnosis, SBO external retention audit | 'w1_diagnosis.py' |
| W2 | LightGBM parameter tuning, cross-validation, model comparison | 'w2_accuracy.py' |
| W3 | Fairlearn constrained training, threshold post-processing, fairness vs. accuracy tradeoff | 'w3_fairness.py' |
| W4 | CTGAN, TVAE, Gaussian Copula, CopulaGAN synthesis and TSTR evaluation | 'w4_synth.py' |
| Expand | Multiple random seeds, sparse population analysis, experimental histogram synthesizer | 'w3_repeated.py'、'w3_sparse.py'、'w4_repeated.py'、'w4_formal_dp.py' |

'TSTR' Train on Synthetic data and Train on Real data; 'TRTR' Train on Synthetic data and Train on Real data

## 目录

```text
github_release/
├── README.md
├── requirements.txt             # Direct dependencies for the complete experiment
├── requirements-tested.txt      # The direct dependency versions used in this verification
├── setup.py
├── research_main.py             # Recommended entry: Run according to the experimental phase
├── main.py                      # Compatible entry: Run according to W1–W4 numbers
├── w1_diagnosis.py ... w4_synth.py
├── w3_repeated.py / w3_sparse.py
├── w4_repeated.py / w4_formal_dp.py
├── formal_dp_synthesizer.py
├── configs/experiment.json      # Data path, features, model, and experimental parameters
├── src/fairness_lab/            # Data processing, model, metrics, and tools
├── scripts/                     # Stage entry, data validation, and result summary
├── tests/                       # Test of metrics, tag leakage, hold-out protocols, etc.
├── data/                        # Raw data, dictionary, source, and validation list
├── results/                     # Existing result snapshots (csV/JsoN)
└── docs/                        # Code navigation, record organization, and verificationinstructions
```

New results generated from the run written to 'artifacts/', which is added to '.gitignore'. Existing result snapshots are saved in 'results/', and running a new experiment will not overwrite them.

## 安装

建议使用 Python 3.10；本次验证环境为 Python 3.10.20。所有命令均在本文件所在目录运行。

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

安装和检查：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/check_env.py
python scripts/verify_data.py
python -m pytest tests -q
```

若 PowerShell 不允许激活脚本，可直接使用 `.\.venv\Scripts\python.exe` 替代以上命令中的 `python`。需要对齐本次验证的依赖版本时，将安装命令中的 `requirements.txt` 换成 `requirements-tested.txt`；后者只固定直接依赖，并非完整环境锁文件。

## 快速开始

先在较小的 Massachusetts 2019 数据上运行基线：

```bash
python research_main.py --stages baseline --datasets acs_ma_2019
```

运行预测模型比较（跳过 Optuna 搜索）：

```bash
python research_main.py --stages model_selection --datasets acs_ma_2019 --skip-optuna
```

运行公平性优化：

```bash
python research_main.py --stages fairness_mitigation --datasets acs_ma_2019
```

只运行较轻量的 Gaussian Copula 合成器：

```bash
python scripts/run_w4_synth.py --datasets acs_ma_2019 --synthesizers gaussian_copula --privacy 0.0
```

运行默认数据集上的全部四阶段：

```bash
python research_main.py
```

完整实验包含 80 次 Optuna 搜索，以及配置为 300 epochs 的神经网络合成器，运行成本明显高于以上单阶段示例。参数在 `configs/experiment.json` 中修改。兼容入口示例为 `python main.py --week 1 2 --datasets acs_ma_2019 --skip-optuna`。

## 扩展运行

```bash
# 在 SBO target 上训练，在 withheld 上进行外部审计
python research_main.py --stages baseline --datasets sbo sbo_withheld

# 三个随机种子的公平性实验
python w3_repeated.py --datasets acs_ma_2019 --seeds 42 43 44

# 稀疏群体分析
python w3_sparse.py --datasets acs_ma_2019 --seeds 42 43 44 --min-group-size 30

# Gaussian Copula 的重复实验
python w4_repeated.py --datasets acs_ma_2019 --seeds 42 43 44 --synthesizers gaussian_copula --privacy-levels 0.0

# 将本次运行的 W1–W4 结果汇总为 Markdown
python scripts/generate_final_report.py
```

GPU 可通过 W4 脚本的 `--use-gpu` 开启，需使用支持 CUDA 的 PyTorch 环境；默认示例使用 CPU。`--out-root` 可以改变主入口的结果目录，但汇总及可视化脚本默认读取根目录下的 `artifacts/`。

## 数据

数据来源是 NIST Data Excerpt Benchmarks。ACS 与 SBO 的官方背景可见 [NIST CRC](https://pages.nist.gov/privacy_collaborative_research_cycle/)，原始说明保存在 [上游 README](data/raw/BenchmarkData/README.md)。本项目使用的是本地保存的数据快照。

| 数据集参数 | 原始行数 | 列数 | 用途 |
| --- | ---: | ---: | --- |
| `acs_ma_2019` | 7,634 | 24 | 训练及测试 |
| `acs_tx_2019` | 9,276 | 24 | 训练及测试 |
| `acs_national_2019` | 27,253 | 24 | 训练及测试 |
| `sbo` | 161,079 | 130 | 训练及测试 |
| `sbo_withheld` | 147,082 | 130 | 推荐入口仅用于 W1 外部审计 |

另附三地区 2018 年 CSV，用作对照数据，未加入默认实验配置。行数按本项目实际 CSV 统计，预处理后的样本量可能因去重、筛选或抽样而变化。数据目录和恢复记录见 [数据说明](data/README.md)。

## 指标与解释

- 效用指标包含 Accuracy、F1、AUC；具体输出列以各阶段结果表为准。
- 公平性指标包括 SPD、EOD、DI、Theil 及交叉群体指标，解释时应同时查看群体样本量与正负例数。
- ACS 标签由 `PINCP > 50000` 构造；SBO 的 `loan_approved` 由 `RECEIPTS_NOISY > 100000` 构造，是规则标签，不是真实贷款审批记录。构造标签的源字段从模型特征中排除。
- W3 对超过 15,000 行的 SBO 数据进行抽样，以控制约束训练的计算量。
- W4 的 `privacy_level` 是数值加噪强度，不是差分隐私预算。`formal_dp_synthesizer.py` 为保留的实验性实现，尚未完成正式隐私保证审计，不能仅凭文件名或输出中的 epsilon 声称严格差分隐私。
- `NaN`、错误列、回退 backend 和未成功生成的组合需要保留并解释。阶段进程正常结束不代表每个实验组合都成功。
- 数据加载器保留旧版 `auto` 演示数据回退行为，因此正式运行前应执行 `verify_data.py`，并检查输出中的 `source`。演示结果不能视为真实数据实验。

## 已有结果与验证

`results/` 包含整理前保存的 W1–W4 表格及 Massachusetts 两种子 W3 实验；这些是历史结果，不代表本次重新运行了全部实验。两种子结果仅适合探索性比较。

整理版验证范围及已知限制见 [验证记录](docs/VALIDATION.md)。代码导航见 [代码说明](docs/CODE_GUIDE.md)。

## 上传到 GitHub

上传此目录中的内容，使 `README.md` 位于仓库根目录。建议仓库名为 `synthetic-data-fairness`。

最大的两个 SBO CSV 约为 64.8 MiB 和 60.6 MiB。GitHub 网页上传限制为单文件 25 MiB，普通 Git 的单文件上限为 100 MiB，所以本项目可使用 Git 命令行或 GitHub Desktop 上传，无需为当前文件配置 Git LFS；超过 50 MiB 的文件可能产生提示。[GitHub 文件大小说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

先在 GitHub 创建一个空仓库，然后在此目录执行以下命令；把 `YOUR_USERNAME` 和仓库名替换为实际值：

```bash
git init
git add .
git commit -m "Initial release: code, benchmark data and results"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/synthetic-data-fairness.git
git push -u origin main
```

数据的来源说明和贡献者信息保留在上游 README 中。本整理版未替项目作者选择代码开源许可证。
