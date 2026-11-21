"""MMLongBench-Doc (NeurIPS 2024) Multimodal Long-PDF Benchmark Runner for AnchorDoc.

Evaluates 101 official questions across 7 non-financial multimodal PDFs (research reports,
slide decks, crisis analyses, surveys — heavy Charts, Figures, Tables, and Unanswerable items).

Compares 3 methods on the exact same multimodal model (deepseek-v4-flash, temperature=0):
  1. ChunkRAG-TextOnly       (flat 1000-char text chunks, blind to charts/figures)
  2. AnchorDoc-TextOnly      (AnchorMap + C3 + B42, text-only QA)
  3. AnchorDoc-Multimodal    (AnchorMap + C3 + B42 + Rendered Original-Page PNG Vision Verification)
"""
from __future__ import annotations

import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.multimodal_doc import MultimodalDocAdapter
from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.runtime import CachedLLM, tokenize


def _parse_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    try:
        x = ast.literal_eval(str(val))
        return x if isinstance(x, list) else []
    except Exception:
        return []


def is_unanswerable(row: dict) -> bool:
    return str(row.get("answer") or "").strip().lower() == "not answerable" or not _parse_list(row.get("evidence_pages"))


def is_visual_only_or_mixed(row: dict) -> bool:
    srcs = " ".join(str(x) for x in _parse_list(row.get("evidence_sources")))
    return ("Chart" in srcs) or ("Figure" in srcs)


def answer_chunk_text(idx: DocumentIndex, question: str, llm: CachedLLM) -> tuple[str, bool]:
    scores = idx.bm_chunk.get_scores(tokenize(question))
    order = sorted(range(len(idx.chunks)), key=lambda i: -scores[i])[:5]
    ctx = "\n\n---\n\n".join(f"[Page {idx.chunks[i][0]}]\n{idx.chunks[i][1]}" for i in order)
    parsed, _ = llm.chat_json(
        "mm_chunk_text",
        "You are a strict document QA reviewer. Answer ONLY from the excerpts. Return JSON only.",
        (
            f"Question: {question}\n\nExcerpts:\n{ctx}\n\n"
            "If the excerpts do not contain the answer, set can_answer=false and answer='Not answerable'.\n"
            'Return JSON: {"can_answer": true/false, "answer": ""}'
        ),
        max_tokens=300,
    )
    can = bool(parsed.get("can_answer", False))
    ans = str(parsed.get("answer") or "").strip()
    if not can or ans.lower() in {"", "not answerable", "unanswerable"}:
        return "Not answerable", False
    return ans, True


def judge_mm(question: str, gold: str, pred: str, pred_answered: bool, gold_unans: bool, llm: CachedLLM) -> dict[str, Any]:
    if gold_unans:
        # Strict zero-hallucination scoring: correct ONLY if system abstained / said Not answerable!
        correct = (not pred_answered) or ("not answerable" in pred.lower()) or ("unanswerable" in pred.lower())
        return {"correct": correct, "hallucinated_on_unans": not correct}
    if not pred_answered or not pred.strip() or pred.strip().lower() == "not answerable":
        return {"correct": False, "abstained_on_ans": True}
    parsed, _ = llm.chat_json(
        "mm_judge",
        "You are a strict benchmark evaluator for MMLongBench-Doc. Return JSON only.",
        (
            f"Question: {question}\n"
            f"Gold Answer: {gold}\n"
            f"Predicted Answer: {pred}\n\n"
            "Is the predicted answer semantically equivalent to the gold answer "
            "(allowing minor formatting/rounding within ~2% for numbers, or equivalent phrasing)?\n"
            'Return JSON: {"correct": true/false}'
        ),
        max_tokens=80,
    )
    return {"correct": bool(parsed.get("correct")), "abstained_on_ans": False}


def summarize(judges: list[dict], rows: list[dict], stats: dict) -> dict[str, Any]:
    n = len(rows)
    tot_corr = sum(1 for j in judges if j["correct"])
    ans_idx = [i for i, r in enumerate(rows) if not is_unanswerable(r)]
    unans_idx = [i for i, r in enumerate(rows) if is_unanswerable(r)]
    vis_idx = [i for i, r in enumerate(rows) if not is_unanswerable(r) and is_visual_only_or_mixed(r)]
    txt_idx = [i for i, r in enumerate(rows) if not is_unanswerable(r) and not is_visual_only_or_mixed(r)]

    def _pct(idxs: list[int]) -> float:
        return round(sum(1 for i in idxs if judges[i]["correct"]) / len(idxs) * 100, 1) if idxs else 0.0

    halluc_unans = sum(1 for i in unans_idx if judges[i].get("hallucinated_on_unans"))
    return {
        "n": n,
        "overall_acc_%": round(tot_corr / n * 100, 1),
        "answerable_acc_%": _pct(ans_idx),
        "visual_chart_fig_acc_%": _pct(vis_idx),
        "text_table_acc_%": _pct(txt_idx),
        "unanswerable_abstain_acc_%": _pct(unans_idx),
        "false_answer_on_unans_%": round(halluc_unans / len(unans_idx) * 100, 1) if unans_idx else 0.0,
        "avg_prompt_tokens": round(stats["prompt_tokens"] / n, 0),
    }


def main() -> None:
    t0 = time.time()
    base = ROOT / "data/mmlongbench"
    all_rows = json.load(open(base / "samples.json", encoding="utf-8"))
    pdf_dir = base / "pdfs"
    avail = {p.name for p in pdf_dir.glob("*.pdf")}
    rows = [r for r in all_rows if r["doc_id"] in avail]
    unique_docs = sorted({r["doc_id"] for r in rows})
    print(f"MMLongBench-Doc Run: {len(rows)} official questions across {len(unique_docs)} multimodal PDFs")

    docs = {d: PdfDocument.from_pdf(pdf_dir / d, backend="pypdfium2") for d in unique_docs}
    indices = {d: DocumentIndex(docs[d]) for d in unique_docs}
    tot_pages = sum(d.num_pages for d in docs.values())
    print(f"Loaded & Zero-LLM indexed {len(docs)} PDFs ({tot_pages} pages)")

    llm = CachedLLM(cache_dir=ROOT / "cache/mmlongbench")

    # 1. ChunkRAG-TextOnly
    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=6) as ex:
        c_out = list(ex.map(lambda r: answer_chunk_text(indices[r["doc_id"]], r["question"], llm), rows))
    c_stats = llm.snapshot_stats()

    # Shared adapters WITHOUT selective vision indexing (v0.3 baseline)
    adapters_novis = {d: MultimodalDocAdapter(docs[d], llm, auto_vision_index=False) for d in unique_docs}
    for d in unique_docs:
        adapters_novis[d].retriever.index = indices[d]

    # 2. AnchorDoc-TextOnly
    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=6) as ex:
        t_out = list(ex.map(lambda ir: adapters_novis[ir[1]["doc_id"]].query(ir[1]["question"], item_id=f"mm_{ir[0]}", top_k=5, use_vision=False), enumerate(rows)))
    t_stats = llm.snapshot_stats()

    # 3. AnchorDoc-v0.4 (Selective VisionPageCard Indexing ONLY on low-text pages + Vision QA + Absence Gate)
    llm.reset_stats()
    t_vidx = time.time()
    adapters_v04 = {d: MultimodalDocAdapter(docs[d], llm, auto_vision_index=True, min_text_chars=180) for d in unique_docs}
    vidx_sec = round(time.time() - t_vidx, 1)
    vidx_pages = sum(a.vision_index_stats["vision_indexed_pages"] for a in adapters_v04.values())
    print(f"Selective VisionPageCard indexed {vidx_pages}/{tot_pages} low-text pages in {vidx_sec}s (text-rich pages skipped = {tot_pages - vidx_pages})")

    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=6) as ex:
        v_out = list(ex.map(lambda ir: adapters_v04[ir[1]["doc_id"]].query(ir[1]["question"], item_id=f"mm_{ir[0]}", top_k=5, use_vision=True), enumerate(rows)))
    v_stats = llm.snapshot_stats()

    # Evidence page Hit@5 on answerable questions (Chunk-BM25 vs v0.3 Text-Index vs v0.4 Selective-Vision-Index)
    ans_rows = [(r, t, v) for r, t, v in zip(rows, t_out, v_out) if not is_unanswerable(r)]
    chunk_hit5 = sum(
        1 for r, _, _ in ans_rows
        if set(int(x) for x in _parse_list(r["evidence_pages"])) & set(indices[r["doc_id"]].rank_chunk_bm25_pages(r["question"], top_k=5))
    ) / len(ans_rows) * 100
    v03_hit5 = sum(
        1 for r, t, _ in ans_rows
        if set(int(x) for x in _parse_list(r["evidence_pages"])) & set(t.retrieval_trace.get("pages", [])[:5])
    ) / len(ans_rows) * 100
    ret_hit5 = sum(
        1 for r, _, v in ans_rows
        if set(int(x) for x in _parse_list(r["evidence_pages"])) & set(v.retrieval_trace.get("pages", [])[:5])
    ) / len(ans_rows) * 100

    # Grade all 3 methods
    with ThreadPoolExecutor(max_workers=6) as ex:
        jc = list(ex.map(lambda ro: judge_mm(ro[0]["question"], ro[0]["answer"], ro[1][0], ro[1][1], is_unanswerable(ro[0]), llm), zip(rows, c_out)))
        jt = list(ex.map(lambda ro: judge_mm(ro[0]["question"], ro[0]["answer"], ro[1].answer, ro[1].status == "answered", is_unanswerable(ro[0]), llm), zip(rows, t_out)))
        jv = list(ex.map(lambda ro: judge_mm(ro[0]["question"], ro[0]["answer"], ro[1].answer, ro[1].status == "answered", is_unanswerable(ro[0]), llm), zip(rows, v_out)))

    res = {
        "benchmark": f"MMLongBench-Doc ({len(rows)} official questions, {len(docs)} non-financial multimodal PDFs, {tot_pages} pages)",
        "model": llm.model,
        "selective_vision_indexing": {
            "total_pages": tot_pages,
            "vision_indexed_pages": vidx_pages,
            "skipped_text_rich_pages": tot_pages - vidx_pages,
            "wall_sec": vidx_sec,
        },
        "retrieval_hit5_on_answerable": {
            "Chunk-BM25": round(chunk_hit5, 1),
            "AnchorDoc-TextIndex (v0.3)": round(v03_hit5, 1),
            "AnchorDoc-SelectiveVision (v0.4)": round(ret_hit5, 1),
        },
        "methods": {
            "ChunkRAG-TextOnly": summarize(jc, rows, c_stats),
            "AnchorDoc-TextOnly": summarize(jt, rows, t_stats),
            "AnchorDoc-v0.4 (SelectiveVision+AbsenceGate)": summarize(jv, rows, v_stats),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = ROOT / "results/mmlongbench_101_v04.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== MMLongBench-Doc Retrieval Hit@5 (Answerable n={len(ans_rows)}) ===")
    print(f"  Chunk-BM25: {chunk_hit5:.1f}%  |  v0.3 TextIndex: {v03_hit5:.1f}%  |  v0.4 SelectiveVision: {ret_hit5:.1f}%")
    print("\n=== MMLongBench-Doc End-to-End Multimodal Results ===")
    for name, m in res["methods"].items():
        print(
            f"  {name:30s} | Overall={m['overall_acc_%']:5.1f}% | Chart/Fig={m['visual_chart_fig_acc_%']:5.1f}% | "
            f"Text/Tbl={m['text_table_acc_%']:5.1f}% | UnansAbstain={m['unanswerable_abstain_acc_%']:5.1f}% "
            f"(Halluc={m['false_answer_on_unans_%']:4.1f}%)"
        )
    print(f"Saved: {out_path} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
