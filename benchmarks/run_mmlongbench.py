"""MMLongBench-Doc (NeurIPS 2024) Multimodal Long-PDF Benchmark Runner for AnchorDoc.

Evaluates 101 official questions across 7 non-financial multimodal PDFs (research reports,
slide decks, crisis analyses, surveys — heavy Charts, Figures, Tables, and Unanswerable items).

Compares 3 methods on the exact same multimodal model (deepseek-v4-flash, temperature=0):
  1. ChunkRAG-TextOnly       (flat 1000-char text chunks, blind to charts/figures)
  2. AnchorDoc-TextOnly      (AnchorMap + C3 + B42, text-only QA)
  3. AnchorDoc-Multimodal    (AnchorMap + C3 + B42 + Rendered Original-Page PNG Vision Verification)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
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


def _pages_hit(gold: list[int], pages: list[int]) -> tuple[bool, bool]:
    gs = set(gold)
    return bool(gs & set(pages)), bool(gs) and gs.issubset(set(pages))


def main() -> None:
    ap = argparse.ArgumentParser(description="MMLongBench-Doc runner (per-item results, isolated cache/out).")
    ap.add_argument("--cache-dir", default=os.environ.get("ANCHORDOC_BENCH_CACHE", str(ROOT / "cache/mmlongbench")))
    ap.add_argument("--out", default=os.environ.get("ANCHORDOC_BENCH_OUT", str(ROOT / "results/mmlongbench_101_v06.json")))
    ap.add_argument("--methods", default="chunk,text,vision", help="comma list: chunk,text,vision")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}

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

    llm = CachedLLM(cache_dir=args.cache_dir)
    golds = [[int(x) for x in _parse_list(r["evidence_pages"])] for r in rows]
    ans_idx = [i for i, r in enumerate(rows) if not is_unanswerable(r)]
    res: dict[str, Any] = {
        "benchmark": f"MMLongBench-Doc ({len(rows)} official questions, {len(docs)} non-financial multimodal PDFs, {tot_pages} pages)",
        "model": llm.model,
        "retrieval_on_answerable": {},
        "methods": {},
        "per_item": {},
    }
    jobs: dict[str, tuple[list[Any], dict]] = {}

    if "chunk" in wanted:
        llm.reset_stats()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            c_out = list(ex.map(lambda r: answer_chunk_text(indices[r["doc_id"]], r["question"], llm), rows))
        c_stats = llm.snapshot_stats()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            jc = list(ex.map(lambda ro: judge_mm(ro[0]["question"], ro[0]["answer"], ro[1][0], ro[1][1], is_unanswerable(ro[0]), llm), zip(rows, c_out)))
        res["methods"]["ChunkRAG-TextOnly"] = summarize(jc, rows, c_stats)
        chunk_hit = [_pages_hit(golds[i], indices[rows[i]["doc_id"]].rank_chunk_bm25_pages(rows[i]["question"], top_k=5))[0] for i in ans_idx]
        res["retrieval_on_answerable"]["Chunk-BM25"] = {"hit@5": round(sum(chunk_hit) / len(ans_idx) * 100, 1)}

    def _run_adapter(name: str, adapters: dict, use_vision: bool) -> None:
        llm.reset_stats()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            out = list(ex.map(
                lambda ir: adapters[ir[1]["doc_id"]].query(ir[1]["question"], item_id=f"mm_{ir[0]}", top_k=5, use_vision=use_vision),
                enumerate(rows),
            ))
        stats = llm.snapshot_stats()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            js = list(ex.map(
                lambda ro: judge_mm(ro[0]["question"], ro[0]["answer"], ro[1].answer, ro[1].status == "answered", is_unanswerable(ro[0]), llm),
                zip(rows, out),
            ))
        res["methods"][name] = summarize(js, rows, stats)
        res["methods"][name]["avg_calls"] = round(stats["calls"] / len(rows), 2)
        h5 = [_pages_hit(golds[i], out[i].retrieval_trace.get("pages", [])[:5]) for i in ans_idx]
        read = [_pages_hit(golds[i], (out[i].meta or {}).get("rendered_pages") or out[i].retrieval_trace.get("pages", [])[:5]) for i in ans_idx]
        res["retrieval_on_answerable"][name] = {
            "hit@5": round(sum(h for h, _ in h5) / len(ans_idx) * 100, 1),
            "all_gold@5": round(sum(a for _, a in h5) / len(ans_idx) * 100, 1),
            "hit@read_pages": round(sum(h for h, _ in read) / len(ans_idx) * 100, 1),
            "all_gold@read_pages": round(sum(a for _, a in read) / len(ans_idx) * 100, 1),
        }
        res["per_item"][name] = [
            {
                "i": i,
                "doc": r["doc_id"],
                "question": r["question"],
                "gold": r["answer"],
                "answer_format": r.get("answer_format"),
                "evidence_sources": r.get("evidence_sources"),
                "unanswerable": is_unanswerable(r),
                "visual": is_visual_only_or_mixed(r),
                "gold_pages": golds[i],
                "pages": v.retrieval_trace.get("pages", []),
                "status": v.status,
                "pred": v.answer[:400],
                "rule_trace": v.rule_trace,
                "correct": bool(j.get("correct")),
                "meta": v.meta,
            }
            for i, (r, v, j) in enumerate(zip(rows, out, js))
        ]

    if "text" in wanted:
        adapters_novis = {d: MultimodalDocAdapter(docs[d], llm, auto_vision_index=False) for d in unique_docs}
        for d in unique_docs:
            adapters_novis[d].retriever.index = indices[d]
        _run_adapter("AnchorDoc-TextOnly", adapters_novis, use_vision=False)

    if "vision" in wanted:
        llm.reset_stats()
        t_vidx = time.time()
        adapters_vis = {d: MultimodalDocAdapter(docs[d], llm, auto_vision_index=True, min_text_chars=180) for d in unique_docs}
        vidx_sec = round(time.time() - t_vidx, 1)
        vidx_pages = sum(a.vision_index_stats["vision_indexed_pages"] for a in adapters_vis.values())
        res["selective_vision_indexing"] = {
            "total_pages": tot_pages,
            "vision_indexed_pages": vidx_pages,
            "skipped_text_rich_pages": tot_pages - vidx_pages,
            "wall_sec": vidx_sec,
            "indexing_stats": llm.snapshot_stats(),
        }
        print(f"Selective VisionPageCard indexed {vidx_pages}/{tot_pages} low-text pages in {vidx_sec}s")
        _run_adapter("AnchorDoc-Full (SelectiveVision+EvidenceGate)", adapters_vis, use_vision=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== MMLongBench-Doc Retrieval on Answerable (n={len(ans_idx)}) ===")
    for name, m in res["retrieval_on_answerable"].items():
        print(f"  {name:48s} | {m}")
    print("\n=== MMLongBench-Doc End-to-End Results ===")
    for name, m in res["methods"].items():
        print(
            f"  {name:48s} | Overall={m['overall_acc_%']:5.1f}% | Chart/Fig={m['visual_chart_fig_acc_%']:5.1f}% | "
            f"Text/Tbl={m['text_table_acc_%']:5.1f}% | UnansAbstain={m['unanswerable_abstain_acc_%']:5.1f}% "
            f"(Halluc={m['false_answer_on_unans_%']:4.1f}%) | AvgTok={int(m['avg_prompt_tokens'])}"
        )
    print(f"Saved: {out_path} ({res['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
