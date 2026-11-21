<div align="center">

# ⚓ AnchorDoc

**面向 100–600 页高风险长文档的零建索引成本、无向量可核验审计引擎**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](pyproject.toml)
[![FinanceBench](https://img.shields.io/badge/FinanceBench%20150%2F150-78.7%25%20E2E%20%7C%2090%25%20数值表-16a34a.svg)](benchmarks/LEADERBOARD.md)
[![MMLongBench](https://img.shields.io/badge/MMLongBench--Doc-95.4%25%20Hit%405-2563eb.svg)](benchmarks/LEADERBOARD.md)

[English](README.md) | **简体中文**

</div>

---

**AnchorDoc** 摒弃脆弱的固定切块向量 RAG 与昂贵的全书 LLM 预建树，提出 **“50ms 零 LLM 全文结构锚点图（AnchorMap） + 印刷目录页偏校准（G7） + 选择性视觉页卡（仅对低字数/扫描图页触发） + 两步缺项补看（B42） + 事实与裁决解耦”** 架构，以纯 Apache-2.0 内核实现高准确率、极低建索引成本与页级可审计溯源。

![AnchorDoc 架构图](assets/architecture.svg)

---

## 核心对比（统一底座 `deepseek-v4-flash`, `temperature=0`, Apache-2.0 `pypdfium2`）

| 方案 | 建索引成本（250 页 PDF） | FinanceBench 全量 150 题 E2E | 多页证据找全 (AllGold@5) | 纯图/扫描 PPT 表现 |
|---|---:|---:|---:|---|
| 传统平铺切块 BM25 | `0.3s` · `0` 次 LLM | `33.3%` | `32.0%` | 不可读 |
| Dense 向量 RAG（`BGE-small` ONNX） | `47.6s`（向量推理） | `39.3%` | `48.7%` | 不可读 |
| LLM 预建树（PageIndex 风格 12 页窗口） | `11.4s` · `19.2` 次 · `33k` tok | `35.9%` *（39 题子集）* | `43.6%` *（39 题子集）* | 需逐页调 VLM |
| **⚓ AnchorDoc `v0.5`** | **`0.32s` · `0` 次 LLM** | **`78.7%`（数值表 `90.0%`）** | **`81.3%`（数值表 `100%`）** | **选择性视觉页卡（`95.4%` Hit@5）** |

![AnchorDoc 公开基准对比](assets/benchmark_comparison.svg)

---

## 三大国际公开基准实测（401 道官方题，全量可复现）

1. **FinanceBench 官方全量 150 / 150（84 份真实 SEC 年报/季报，12,013 页）**
   - 建索引耗时 **26.7 秒（`0.32s/份`），0 次 LLM 调用**（比 CPU Dense BGE 快 **149 倍**，比 LLM 预建树快 **91 倍**）；
   - 页级召回 **Hit@5 `84.7%` / AllGold@5 `81.3%`**（Dense BGE 仅 `58.0% / 48.7%`）；消融证实 `AnchorMap + G7` 单独带来 **+18.0pp AllGold@5**；
   - 端到端准确率 **`78.7%`**（Dense BGE `39.3%` 的 **2.0 倍**）；全部 50 道硬数值报表题 **`100.0%` 找全 / `90.0%` 端到端准确**（ChunkRAG 仅 `20.0%`）。
2. **MMLongBench-Doc（NeurIPS 2024，7 份非金融多模态 PDF 含 3 份 0 字纯图 PPT，101 题）**
   - **`SelectiveVisionIndexer`** 自动跳过全部 78 页文字页（0 成本）、仅对 92 页低字数/纯图页构建视觉页卡，把可回答题 **Hit@5 从 `63.1%` 推升至 `95.4%`（62/65）**（Chunk-BM25 仅 `44.6%`）；
   - 端到端总准 **`52.5%`**（ChunkRAG `36.6%`），文本表格 **`53.1%`**（vs `21.9%`），图表 **`24.2%`**（vs `0.0%`），不可回答题拒答准确率 **`77.8%`**（严格视觉判空门禁将幻觉率近乎腰斩）。
3. **ContractNLI-Long（150 条 ≥1 万字长法律合同，三态均衡 50/50/50）**
   - 在 Macro-F1（`78.1%` vs `78.0%`）、高置信度准确率（`80.3%` vs `78.5%`）、误通过率（`14.0%` vs `16.0%`）上全面帕累托占优“整本硬塞 Long-Context”，同时**节省 25% Prompt Token**。

完整榜单与复现脚本见 [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md)。

---

## 60 秒快速上手

```bash
git clone https://github.com/WuHuiYang/AnchorDoc.git
cd AnchorDoc
pip install -e .

# 1. 无需任何 API Key：0.2 秒查看全文结构锚点图与印刷目录页码偏移
anchordoc outline annual_report_10k.pdf

# 2. 端到端可核验问答 + 一键导出“左结论、右原页截图”双栏单文件 HTML 报告
export ANCHORDOC_LLM_BASE=https://api.openai.com/v1
export ANCHORDOC_LLM_MODEL=gpt-4o-mini
export ANCHORDOC_LLM_KEY=sk-...

anchordoc query annual_report_10k.pdf "What is FY2023 gross profit margin?" --html report.html

# 3. 按 JSON 清单审计长 PDF（扫描件/纯图 PDF 加 --vision）
anchordoc audit submission.pdf --checklist examples/sample_checklist.json --html audit_workbench.html
```

---

## 开源协议与引用

本项目采用商业友好的 **Apache-2.0** 协议（主包零 AGPL 强依赖）。如果本项目对你的研究或工程有帮助，欢迎通过 [CITATION.cff](CITATION.cff) 引用。
