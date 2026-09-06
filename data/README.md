# 数据说明

`raw/BenchmarkData/` 保存本项目使用的 NIST ACS / SBO 数据快照、JSON 数据字典、映射和元数据。数据背景见 [NIST CRC](https://pages.nist.gov/privacy_collaborative_research_cycle/)；原始说明与署名见 [上游 README](raw/BenchmarkData/README.md)。

## 文件布局

| 子目录 | CSV | 原始行数 |
| --- | --- | ---: |
| `ACSDataExcerpts/massachusetts` | `ma2018.csv` / `ma2019.csv` | 7,244 / 7,634 |
| `ACSDataExcerpts/texas` | `tx2018.csv` / `tx2019.csv` | 8,775 / 9,276 |
| `ACSDataExcerpts/national` | `national2018.csv` / `national2019.csv` | 27,111 / 27,253 |
| `SBODataExcerpts` | `sbo_target.csv` / `sbo_withheld.csv` | 161,079 / 147,082 |

ACS 每份 CSV 为 24 列；SBO 为 130 列。`config.json`、`data_dictionary.json`、`mappings.json` 及与 CSV 同名的 JSON 均为上游数据辅助文件。`configs/experiment.json` 才是本项目的运行配置。

ACS 使用字面值 `N` 表示部分缺失值，项目配置已经声明该规则。SBO 的缺失值通常是 CSV 空字段。

## 整理记录

- 保留所有八份独立 CSV，CSV 数据内容未重新编码或改写。
- 原目录的 `ma2018.csv` 只有 Git LFS 指针。本版从原有 `BenchmarkData.zip` 恢复实际 CSV，并确认其 SHA-256 与指针一致：`36efe30faec9c83b60317927f762db745241a856353205f8b5a47facab40980e`。
- 移除 ACS 根目录与地区目录重复的 2019 CSV、WPS 同步副本、`__MACOSX`、重复压缩包，以及无实际 PDF 内容的 postcard 指针文件。
- 保留上游 README 原文；其中 national 2019 的描述行数与当前快照不同。本项目实际统计为 27,253 行，以 CSV 与校验清单为准。
- 原始数据以外的已有实验结果放在 `results/`。原项目没有保存可交付的 W4 合成样本 CSV；W4 主流程默认保存评估表，合成表在运行时生成。

## 完整性校验

在项目根目录运行：

```bash
python scripts/verify_data.py
```

该脚本仅使用 Python 标准库，核对 `manifest.json` 中的 SHA-256、字节数、CSV 行列数及默认配置中的数据路径。`.gitattributes` 禁用原始数据的换行转换，以保证 Windows 与其他平台的文件校验一致。
