"""Full 150-question official FinanceBench benchmark runner for AnchorDoc.

Evaluates ALL 150 official open-source FinanceBench questions across all 84 real SEC
10-K / 10-Q / 8-K filings using Apache-2.0 pypdfium2 + AnchorDoc core:
  - Page-Level Retrieval: Chunk-BM25 vs RawPage-BM25 vs AnchorDoc (Zero-LLM AnchorMap + G7 + C3 + B42)
  - End-to-End Verifiable QA: ChunkRAG-E2E vs AnchorDoc-E2E (verbatim grounding + Python AST calc)
  - Zero-LLM Indexing Speed & Cost Accounting (wall-clock sec/PDF + LLM tokens/question)

Usage:
  ANCHORDOC_LLM_BASE=... ANCHORDOC_LLM_MODEL=... ANCHORDOC_LLM_KEY_FILE=... \
    python3 benchmarks/run_financebench_full.py [--limit 150] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.finance_10k import Finance10KAdapter
from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.runtime import CachedLLM, tokenize


def resolve_gold_pages(doc: PdfDocument, ev_list: list[dict]) -> list[int]:
    """FinanceBench uses 0-based evidence_page_num -> 1-based physical PDF page."""
    gold: list[int] = []
    for ev in ev_list or []:
        raw_p = ev.get("evidence_page_num")
        if isinstance(raw_p, int):
            p = raw_p + 1
            if 1 <= p <= doc.num_pages and p not in gold:
                gold.append(p)
    return gold


def retrieval_metrics(preds_list: list[list[int]], golds_list: list[list[int]]) -> dict[str, Any]:
    n = len(golds_list)
    r1 = r3 = r5 = all5 = 0
    mrr_sum = 0.0
    for preds, golds in zip(preds_list, golds_list):
        gset = set(golds)
        if not gset:
            continue
        if any(p in gset for p in preds[:1]):
            r1 += 1
        if any(p in gset for p in preds[:3]):
            r3 += 1
        if any(p in gset for p in preds[:5]):
            r5 += 1
        if gset.issubset(set(preds[:5])):
            all5 += 1
        rr = 0.0
        for rank, p in enumerate(preds[:10], start=1):
            if p in gset:
                rr = 1.0 / rank
                break
        mrr_sum += rr
    return {
        "n": n,
        "hit@1": round(r1 / n * 100, 1) if n else 0.0,
        "hit@3": round(r3 / n * 100, 1) if n else 0.0,
        "hit@5": round(r5 / n * 100, 1) if n else 0.0,
        "all_gold@5": round(all5 / n * 100, 1) if n else 0.0,
        "mrr@10": round(mrr_sum / n, 4) if n else 0.0,
    }


def answer_chunk_rag_e2e(idx: DocumentIndex, question: str, llm: CachedLLM) -> dict[str, Any]:
    scores = idx.bm_chunk.get_scores(tokenize(question))
    order = sorted(range(len(idx.chunks)), key=lambda i: -scores[i])[:5]
    ctx = "\n\n---\n\n".join(f"[Page {idx.chunks[i][0]}]\n{idx.chunks[i][1]}" for i in order)
    parsed, _ = llm.chat_json(
        "fb150_chunk_e2e",
        "You are a financial analyst answering questions strictly from SEC filing excerpts. Return JSON only.",
        (
            f"Question: {question}\n\nExcerpts:\n{ctx}\n\n"
            "Answer concisely based ONLY on the excerpts. Set can_answer=false if insufficient.\n"
            'Return JSON: {"can_answer": true/false, "answer": "", "cited_pages": [int]}'
        ),
        max_tokens=600,
    )
    return {
        "can_answer": bool(parsed.get("can_answer", True)),
        "answer": str(parsed.get("answer") or "").strip(),
    }


def judge_answer(question: str, gold: str, pred: str, can_answer: bool, llm: CachedLLM) -> dict[str, Any]:
    if not can_answer or not pred.strip():
        return {"verdict": "abstain", "note": "empty_or_abstained"}
    parsed, _ = llm.chat_json(
        "fb150_judge",
        "You are a strict financial QA evaluator. Return JSON only.",
        (
            f"Question: {question}\n"
            f"Official Gold Answer: {gold}\n"
            f"Predicted Answer: {pred}\n\n"
            "Grade strictly:\n"
            "- Numerical: 'correct' only if the core figure matches gold within ~2% relative tolerance "
            "(allowing rounding or equivalent units like $1.58B vs $1,577M, 24.5% vs 0.245).\n"
            "- Yes/No or qualitative: 'correct' only if the main conclusion and key drivers align with gold.\n"
            "- If predicted answer states information is missing / cannot answer, grade 'abstain'.\n"
            'Return JSON: {"verdict": "correct|incorrect|abstain", "note": ""}'
        ),
        max_tokens=150,
    )
    v = str(parsed.get("verdict") or "incorrect").strip().lower()
    if v not in {"correct", "incorrect", "abstain"}:
        v = "incorrect"
    return {"verdict": v, "note": str(parsed.get("note") or "")}


def summarize_qa(judges: list[dict], stats: dict) -> dict[str, Any]:
    n = len(judges)
    corr = sum(1 for j in judges if j["verdict"] == "correct")
    inc = sum(1 for j in judges if j["verdict"] == "incorrect")
    abst = sum(1 for j in judges if j["verdict"] == "abstain")
    answered = corr + inc
    return {
        "n": n,
        "e2e_acc_%": round(corr / n * 100, 1) if n else 0.0,
        "incorrect_%": round(inc / n * 100, 1) if n else 0.0,
        "abstain_%": round(abst / n * 100, 1) if n else 0.0,
        "answered_precision_%": round(corr / answered * 100, 1) if answered else 0.0,
        "avg_prompt_tokens": round(stats["prompt_tokens"] / n, 0) if n else 0,
        "avg_calls": round(stats["calls"] / n, 2) if n else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/financebench_merged.jsonl"))
    ap.add_argument("--pdf-dir", default=str(ROOT / "data/financebench_pdfs"))
    ap.add_argument("--cache-dir", default=os.environ.get("ANCHORDOC_BENCH_CACHE", str(ROOT / "cache/financebench_full")))
    ap.add_argument("--out", default=os.environ.get("ANCHORDOC_BENCH_OUT", str(ROOT / "results/financebench_150_v06.json")))
    ap.add_argument("--backend", default="pypdfium2", choices=["pypdfium2", "pymupdf"])
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    fb = [json.loads(ln) for ln in open(args.data, encoding="utf-8")]
    pdf_dir = Path(args.pdf_dir)
    qs = [r for r in fb if (pdf_dir / f"{r['doc_name']}.pdf").is_file()][: args.limit]
    unique_docs = sorted({q["doc_name"] for q in qs})
    print(f"FinanceBench Full Run: {len(qs)} questions across {len(unique_docs)} real SEC PDFs (backend={args.backend})")

    # Load and build Zero-LLM DocumentIndex for every PDF
    t_idx0 = time.time()
    docs: dict[str, PdfDocument] = {}
    indices: dict[str, DocumentIndex] = {}

    def _load_one(name: str) -> tuple[str, PdfDocument, DocumentIndex]:
        d = PdfDocument.from_pdf(pdf_dir / f"{name}.pdf", backend=args.backend)
        idx = DocumentIndex(d)
        idx.build_structural_outline()
        return name, d, idx

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(unique_docs)))) as ex:
        for name, d, idx in ex.map(_load_one, unique_docs):
            docs[name] = d
            indices[name] = idx

    idx_sec = time.time() - t_idx0
    total_pages = sum(d.num_pages for d in docs.values())
    print(
        f"Indexed {len(docs)} PDFs ({total_pages} pages) in {idx_sec:.1f}s "
        f"({idx_sec / max(1, len(docs)):.2f}s/PDF, 0 indexing LLM calls!)"
    )

    golds = [resolve_gold_pages(docs[q["doc_name"]], q["evidence"]) for q in qs]

    # Zero-LLM Baselines
    p_chunk = [indices[q["doc_name"]].rank_chunk_bm25_pages(q["question"], top_k=10) for q in qs]
    p_raw = [
        [i + 1 for i in sorted(range(indices[q["doc_name"]].n), key=lambda i: -indices[q["doc_name"]].bm_raw.get_scores(tokenize(q["question"]))[i])[:10]]
        for q in qs
    ]
    p_card0 = [indices[q["doc_name"]].rank_pagecard_rrf(q["question"], top_k=10) for q in qs]

    llm = CachedLLM(cache_dir=args.cache_dir)

    # Baseline E2E: ChunkRAG-E2E
    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        chunk_ans = list(ex.map(lambda q: answer_chunk_rag_e2e(indices[q["doc_name"]], q["question"], llm), qs))
    chunk_stats = llm.snapshot_stats()

    # AnchorDoc Full (Retrieval + Fact-Decoupled Numerical/Qualitative QA)
    adapters = {name: Finance10KAdapter(docs[name], llm) for name in unique_docs}
    # Reuse pre-built indices to avoid double indexing
    for name in unique_docs:
        adapters[name].retriever.index = indices[name]

    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ad_verdicts = list(ex.map(lambda q: adapters[q["doc_name"]].query(q["question"], item_id=q["financebench_id"], top_k=5), qs))
    ad_stats = llm.snapshot_stats()
    p_anchor = [v.retrieval_trace.get("pages", []) for v in ad_verdicts]

    # Impartial rubric grading
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        j_chunk = list(ex.map(lambda qa: judge_answer(qa[0]["question"], qa[0]["answer"], qa[1]["answer"], qa[1]["can_answer"], llm), zip(qs, chunk_ans)))
        j_anchor = list(ex.map(lambda qv: judge_answer(qv[0]["question"], qv[0]["answer"], qv[1].answer, qv[1].status == "answered", llm), zip(qs, ad_verdicts)))

    ret_table = {
        "Chunk-BM25 (flat 1000ch)": retrieval_metrics(p_chunk, golds),
        "RawPage-BM25": retrieval_metrics(p_raw, golds),
        "AnchorDoc-PageCard (zero-LLM)": retrieval_metrics(p_card0, golds),
        "AnchorDoc-Full (AnchorMap+G7+C3+B42)": retrieval_metrics(p_anchor, golds),
    }
    qa_table = {
        "ChunkRAG-E2E": summarize_qa(j_chunk, chunk_stats),
        "AnchorDoc-E2E": summarize_qa(j_anchor, ad_stats),
    }

    # Per-type breakdown
    by_type: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs):
        by_type[q["question_type"]].append(i)
    type_breakdown = {}
    for qt, idxs in sorted(by_type.items()):
        type_breakdown[qt] = {
            "n": len(idxs),
            "retrieval": retrieval_metrics([p_anchor[i] for i in idxs], [golds[i] for i in idxs]),
            "chunk_e2e_acc_%": round(sum(1 for i in idxs if j_chunk[i]["verdict"] == "correct") / len(idxs) * 100, 1),
            "anchordoc_e2e_acc_%": round(sum(1 for i in idxs if j_anchor[i]["verdict"] == "correct") / len(idxs) * 100, 1),
            "anchordoc_abstain_%": round(sum(1 for i in idxs if j_anchor[i]["verdict"] == "abstain") / len(idxs) * 100, 1),
        }

    out_obj = {
        "benchmark": f"FinanceBench Official Open-Source ({len(qs)} questions, {len(docs)} PDFs, {total_pages} pages)",
        "model": llm.model,
        "pdf_backend": args.backend,
        "indexing": {
            "total_pdfs": len(docs),
            "total_pages": total_pages,
            "wall_sec": round(idx_sec, 2),
            "sec_per_pdf": round(idx_sec / max(1, len(docs)), 3),
            "indexing_llm_calls": 0,
        },
        "retrieval": ret_table,
        "e2e_qa": qa_table,
        "by_question_type": type_breakdown,
        "per_item": [
            {
                "id": q["financebench_id"],
                "doc": q["doc_name"],
                "type": q["question_type"],
                "gold_pages": g,
                "anchor_pages": pa[:5],
                "chunk_verdict": jc["verdict"],
                "anchor_verdict": ja["verdict"],
                "quote_verified": v.quote_verified,
                "calc_used": v.calc_value is not None,
                "anchor_answer": v.answer[:400],
                "gold_answer": str(q["answer"])[:180],
                "rule_trace": v.rule_trace,
                "meta": v.meta,
                "retrieval_layers": v.retrieval_trace.get("layers", {}),
                "followup_triggered": v.retrieval_trace.get("followup_triggered", False),
            }
            for q, g, pa, jc, ja, v in zip(qs, golds, p_anchor, j_chunk, j_anchor, ad_verdicts)
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== FinanceBench FULL Retrieval Results ===")
    for name, m in ret_table.items():
        print(f"  {name:38s} | Hit@1={m['hit@1']:5.1f}% | Hit@3={m['hit@3']:5.1f}% | Hit@5={m['hit@5']:5.1f}% | AllGold@5={m['all_gold@5']:5.1f}% | MRR={m['mrr@10']:.4f}")
    print("\n=== FinanceBench FULL End-to-End QA Results ===")
    for name, m in qa_table.items():
        print(f"  {name:38s} | Acc={m['e2e_acc_%']:5.1f}% | Wrong={m['incorrect_%']:5.1f}% | Abstain={m['abstain_%']:5.1f}% | AnsweredPrec={m['answered_precision_%']:5.1f}% | AvgTok={int(m['avg_prompt_tokens'])}")
    print("\n=== Breakdown by Question Type ===")
    for qt, b in type_breakdown.items():
        r = b["retrieval"]
        print(f"  {qt:20s} (n={b['n']:2d}) | Hit@5={r['hit@5']:5.1f}% | AllGold@5={r['all_gold@5']:5.1f}% | ChunkE2E={b['chunk_e2e_acc_%']:5.1f}% | AnchorDocE2E={b['anchordoc_e2e_acc_%']:5.1f}% (Abst={b['anchordoc_abstain_%']:4.1f}%)")
    print(f"\nSaved: {out_path} ({out_obj['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
