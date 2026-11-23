<div align="center">

# ⚓ AnchorDoc

**面向 100–600 页高风险长文档的零建索引成本、无向量可核验审计引擎**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24453581.svg)](https://doi.org/10.5281/zenodo.24453581)
[![FinanceBench](https://img.shields.io/badge/FinanceBench%20150%2F150-85.3%25%20E2E%20%7C%2096%25%20数值表-16a34a.svg)](benchmarks/LEADERBOARD.md)
[![MMLongBench](https://img.shields.io/badge/MMLongBench--Doc-68.3%25%20E2E%20%7C%2054.5%25%20图表-2563eb.svg)](benchmarks/LEADERBOARD.md)
[![ContractNLI](https://img.shields.io/badge/ContractNLI--Long-81.3%25%20F1%20%7C%2010%25%20FER-7c3aed.svg)](benchmarks/LEADERBOARD.md)
[![MCP](https://img.shields.io/badge/MCP%20Server-零%20LLM%20四工具-0f172a.svg)](#ai-agent-渐进式-mcp-服务端零-api-key)

[English](README.md) | **简体中文**

</div>

---

**AnchorDoc** 摒弃脆弱的固定切块向量 RAG 与昂贵的全书 LLM 预建树，采用 **“0.3s 零 LLM 全文结构锚点图（AnchorMap） + 单次查询感知定位器（Query-Aware Locator） + 选择性视觉页卡与 2×2 高分放大 + 推理优先确定性裁决（Python 权威验算 / 操作数溯源 / 三值视觉门禁 / 两阶段逻辑 NLI）”** 架构，以纯 Apache-2.0 内核实现高准确率、零建索引维护成本与页级可证伪溯源。

![AnchorDoc 演示预览](assets/demo_preview.gif)

![AnchorDoc 架构图](assets/architecture.svg)

---

## 核心对比（统一底座 `deepseek-v4-flash`, `temperature=0`, Apache-2.0 `pypdfium2`）

| 方案 | 建索引成本（250 页 PDF） | FinanceBench 全量 150 题 E2E | 多页证据找全 (AllGold@5) | MMLongBench-Doc (101 题) |
|---|---:|---:|---:|---|
| 传统平铺切块 BM25 | `0.3s` · `0` 次 LLM | `33.3%` | `32.0%` | `36.6%`（图表 `0.0%`） |
| Dense 向量 RAG（`BGE-small` ONNX） | `47.6s`（向量推理） | `39.3%` | `48.7%` | 纯图 PPT 不可读 |
| LLM 预建树（PageIndex 风格 12 页窗口） | `11.4s` · `19.2` 次 · `33k` tok | `35.9%` *（39 题子集）* | `43.6%` *（39 题子集）* | 需逐页调 VLM |
| **⚓ AnchorDoc（`v0.8.0`）** | **`0.33s` · `0` 次 LLM** *（热重载 `3.9ms`）* | **`85.3%`** *（数值表 `96.0%` · `auto` 路由 `84.7%`）* | **`89.3%`** *（数值表 `100%`）* | **`68.3%`** *（图表 `54.5%`）* |

*注：上表统一使用同底座模型（`deepseek-v4-flash`, `temperature=0`）。`v0.8.0` 默认执行路径与冻结基准完全向后兼容，并叠加了 `BundleStore`、前缀 KV 缓存与 4 工具 MCP 服务端；完整逐版本与跨模型（官方 `DeepSeek-V4.1-Flash`）消融明细见 [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md)。*

![AnchorDoc 公开基准对比](assets/benchmark_comparison.svg)

---

## 三大国际公开基准实测（401 道官方题，全量可复现）

1. **FinanceBench 官方全量 150 / 150（84 份真实 SEC 年报/季报，12,013 页）**
   - 建索引耗时 **27.6 秒（`0.33s/份`），0 次 LLM 调用**（比 CPU Dense BGE 快 **145 倍**，比 LLM 预建树快 **91 倍**）；
   - **`v0.6` 单次查询感知定位器**：页级召回跃升至 **Hit@5 `93.3%`（+8.6pp）/ AllGold@5 `89.3%`（+8.0pp）**（Dense BGE 仅 `58.0% / 48.7%`），同时单题 Prompt Token 下降 **23.8%**（`14,150 → 10,780`）、调用次数下降 **22.4%**（`5.35 → 4.15`）；
   - 端到端准确率 **`85.3%`**（**+6.6pp**，配对 McNemar $p=0.0129$，为 Dense BGE `39.3%` 的 **2.17 倍**）；全部 50 道硬数值报表题 **`100.0%` 找全 / `96.0%`（48/50）端到端准确**；在官方 **`DeepSeek-V4.1-Flash`** API 独立复现 **Hit@5 `92.0%` / E2E `82.7%`**（数值报表 `92.0%`，操作数原文溯源 **`101/101`**）。
2. **MMLongBench-Doc（NeurIPS 2024，7 份多模态 PDF 101 题 + 10 份未见盲测 PDF 492 页 91 题）**
   - **`v0.7` `full_pool` + 三值门禁 + 2×2 高分放大**：7 份测试集端到端总准进一步升至 **`68.3%`**，图表问答跃升至 **`54.5%`**（较 v0.5 `24.2%` **提升 2.25 倍**），不可答拒答稳守 **`83.3%`**；
   - **严格未见盲测（`v0.6.0`，10 份全新 PDF 共 492 页，含 52 道图表题）**：检索 Hit@5 **`91.0%`**，图表问答实打实 **`50.0%`**（ChunkRAG 仅 `1.9%`），总准 **`50.5%`**（领先 ChunkRAG +19.7pp）。
3. **ContractNLI-Long（长法律合同 150 题 `test` + 34 份未见合同 150 题 `dev` 盲测）**
   - **`v0.7` 两阶段逻辑 NLI + 自适应 `ReadAll` 路由**：`test-150` 达 **`81.3%` F1 / `10.0%` FER**；在完全未见的 `dev-150` 盲测集上，`route=auto` 在准确率统计持平（`79.3%` vs `78.0%`，Macro-F1 `79.5%` vs `77.9%`）的前提下将 LLM 调用次数直降 **41%**（`~3.0 → 1.77 次/题`）。
4. **文档级 SQLite WAL 持久化（`BundleStore`）与前缀稳定 KV 缓存（`v0.8.0`）**
   - 84 份年报（12,013 页）冷解析 `22.13s` → **热重载仅 `0.330s`（快 `67.1×`，单份 `3.9 ms`）**，且持久化昂贵的 `VisionPageCard` 实现零渲染、零重复 VLM 费用热启动。
   - **T7 3 轮实时方差与真实前缀缓存审计**（30 道多问题共享年报 SEC 10-K 题，官方 `DeepSeek-V4.1-Flash`, `temp=0`）：实时 3 轮 E2E **`85.55% ± 3.85%`**，逐字引文校验 **`100.0% ± 0.0%`**（90/90），前缀稳定 `nav` 触发真实服务端 KV 缓存命中率 **`54.7% ± 10.7%`**（热态最高 **`65.1%`**），离线 SHA 重放方差严格 **`0.00%`**。

---

## AI Agent 渐进式 MCP 服务端（零 API Key）

让 **Claude Code、Cursor、Windsurf、Antigravity、Devin** 等编码 Agent 原生、低 Token、可核验地阅读 500 页 PDF（服务端**默认 0 次外部 LLM 调用**，由宿主 Agent 自己推理，AnchorDoc 担任它的零 LLM 索引、Token 护栏与确定性校验器）：

### 同底座 Agent 实测对比：原生整本塞 PDF vs 接入 AnchorDoc MCP（`DeepSeek-V4.1-Flash`, `temp=0`, 15 份 SEC 10-K/10-Q 财报，均长 144.7 页）

| 同一个 Agent (`DeepSeek-V4.1-Flash`) | 端到端准确率 | 数值报表 / 创新题 | 逐字引文校验率 | 平均总 Prompt Token | 扣缓存后新增计费 Token |
|---|---:|---:|---:|---:|---:|
| 原生整本 PDF 塞上下文 *（450k 字符上限）* | `60.0%` | `80% / 60%` | `53.3%` | `90,824` | `72,827` |
| **⚓ 接入 AnchorDoc 零 LLM MCP** *（`0` 服务端 LLM）* | **`86.7%` (+26.7pp)** | **`100% / 100%`** | **`100.0%` (15/15)** | **`28,188` (-69.0%)** | **`5,430` (-92.5%，省 13.4 倍)** |

*（即便在 8 份 100% 全文未截断的财报上，AnchorDoc MCP 仍以 **`87.5%` vs `50.0%`** 大幅领先，直接规避超长上下文注意力衰减。）*

### 1 分钟接入 Claude Code / Cursor / Windsurf / Devin

```bash
git clone https://github.com/ryanvii/AnchorDoc.git
cd AnchorDoc
pip install -e '.[mcp]'
```

在客户端 `mcp.json`（如 Cursor `.cursor/mcp.json` 或 Claude `claude_desktop_config.json`）中添加：

```json
{
  "mcpServers": {
    "anchordoc": {
      "command": "anchordoc",
      "args": ["mcp", "--root", "/path/to/your/pdfs"]
    }
  }
}
```

仅暴露 4 个硬限流只读正交工具（带 `--root` 目录白名单）：`doc_outline`（L0 地图 ≤1.5k tok）、`doc_locate`（L1 零 LLM 片段定位 ≤1.2k tok）、`doc_read`（L2 最多 5 页 + `p12:L07` 稳定行锚 + 防重复阅读账本 + 原生 MCP `ImageContent`）、`doc_verify`（确定性引文反查 + AST 数学验算 + 操作数溯源）。

### 渐进式上下文工程生态（文档 + 视频双协议联动）

AnchorDoc 与姊妹项目 **🎬 [vidctx-mcp](https://github.com/ryanvii/vidctx-mcp)** 共享同一套 **“零向量库、零服务端 LLM、硬 Token 预算渐进披露”** 协议家族：

| 模态 | 开源仓库 | 渐进式协议链路 | 依赖边界 |
|---|---|---|---|
| **超长 PDF 与高风险文档** | **⚓ [AnchorDoc](https://github.com/ryanvii/AnchorDoc)** *（本仓库）* | `doc_outline` → `doc_locate` → `doc_read` → `doc_verify` | 纯 Python + Apache-2.0 `pypdfium2` |
| **超长视频与屏幕录屏** | **🎬 [vidctx-mcp](https://github.com/ryanvii/vidctx-mcp)** | `peek_video_map` → `search_video_moments` → `inspect_frame_range` | `yt-dlp` / `ffmpeg` / `RapidOCR` |

> **版本复现口径说明**：上表 `v0.6` / `v0.7` / `v0.7.1` 为各阶段冻结基准版本；当前 **`v0.8.0`** 向后兼容保留了核心检索与裁决引擎，并在其上新增了 `BundleStore`、前缀稳定 `nav` 缓存与 4 工具 MCP 服务端（`v0.8.0` 30 题实时 T7 审计在 `DeepSeek-V4.1-Flash` 复现 `85.55% ± 3.85%` E2E 与 `100.0%` 引文校验）。

完整榜单、消融实验、成本摊销分析与复现脚本见 [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md)。

---

## 60 秒快速上手

```bash
git clone https://github.com/ryanvii/AnchorDoc.git
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

本项目采用商业友好的 **Apache-2.0** 协议（主包零 AGPL 强依赖）。如果本项目对你的研究或工程有帮助，欢迎通过 [CITATION.cff](CITATION.cff) 或 Zenodo DOI（`10.5281/zenodo.24453581`）引用：

```bibtex
@software{wu2026anchordoc,
  author  = {Wu, HuiYang},
  title   = {AnchorDoc: Zero-Index-Cost Long-Document Context Engineering with Falsifiable Evidence Chains},
  year    = {2026},
  version = {0.8.0},
  doi     = {10.5281/zenodo.24453581},
  url     = {https://github.com/ryanvii/AnchorDoc},
  license = {Apache-2.0}
}
```
