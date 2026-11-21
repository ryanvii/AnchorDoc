"""Full 150-question FinanceBench Ablation Study for AnchorDoc.

Quantifies the exact causal contribution of each architectural pillar across
ALL 150 official FinanceBench questions (84 SEC PDFs, 12,013 pages):
  1. Chunk-BM25 (flat 1000ch baseline)
  2. AnchorDoc-PageCard (zero-LLM Dual BM25-RRF + TOC filter)
  3. AnchorDoc w/o AnchorMap (C3 + B42 only)
  4. AnchorDoc w/o Gap-Followup (AnchorMap + G7 + C3)
  5. AnchorDoc-Full (AnchorMap + G7 + C3 + B42)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.retrieve import AnchorRetriever
from anchordoc.runtime import CachedLLM
from benchmarks.run_financebench_full import resolve_gold_pages, retrieval_metrics


def main() -> None:
    t0 = time.time()
    fb = [json.loads(ln) for ln in open(ROOT / "data/financebench_merged.jsonl", encoding="utf-8")]
    pdf_dir = ROOT / "data/financebench_pdfs"
    qs = [r for r in fb if (pdf_dir / f"{r['doc_name']}.pdf").is_file()]
    unique_docs = sorted({q["doc_name"] for q in qs})
    print(f"FinanceBench Full Ablation: {len(qs)} questions across {len(unique_docs)} PDFs")

    docs: dict[str, PdfDocument] = {}
    indices: dict[str, DocumentIndex] = {}
    for name in unique_docs:
        d = PdfDocument.from_pdf(pdf_dir / f"{name}.pdf", backend="pypdfium2")
        idx = DocumentIndex(d)
        idx.build_structural_outline()
        docs[name] = d
        indices[name] = idx

    golds = [resolve_gold_pages(docs[q["doc_name"]], q["evidence"]) for q in qs]
    llm = CachedLLM(cache_dir=ROOT / "cache/financebench_full")

    def _run_cfg(label: str, use_anchor: bool, use_fu: bool) -> dict:
        llm.reset_stats()
        retrievers = {}
        for name in unique_docs:
            r = AnchorRetriever(docs[name], llm=llm)
            r.index = indices[name]
            retrievers[name] = r

        with ThreadPoolExecutor(max_workers=8) as ex:
            traces = list(
                ex.map(
                    lambda q: retrievers[q["doc_name"]].retrieve(
                        q["question"],
                        top_k=5,
                        use_anchor_map=use_anchor,
                        use_gap_followup=use_fu,
                    ),
                    qs,
                )
            )
        pages = [tr.pages for tr in traces]
        m = retrieval_metrics(pages, golds)
        st = llm.snapshot_stats()
        fu_pct = round(sum(1 for tr in traces if tr.followup_triggered) / len(traces) * 100, 1)
        m["avg_calls"] = round(st["calls"] / len(qs), 2)
        m["avg_prompt_tokens"] = round(st["prompt_tokens"] / len(qs), 0)
        m["followup_triggered_%"] = fu_pct
        print(
            f"  {label:38s} | Hit@1={m['hit@1']:5.1f}% | Hit@3={m['hit@3']:5.1f}% | "
            f"Hit@5={m['hit@5']:5.1f}% | AllGold@5={m['all_gold@5']:5.1f}% | "
            f"MRR={m['mrr@10']:.4f} | Calls={m['avg_calls']:.2f} | Tok={int(m['avg_prompt_tokens'])}"
        )
        return m

    p_chunk = [indices[q["doc_name"]].rank_chunk_bm25_pages(q["question"], top_k=10) for q in qs]
    p_card = [indices[q["doc_name"]].rank_pagecard_rrf(q["question"], top_k=10) for q in qs]

    results = {
        "benchmark": f"FinanceBench Full Ablation ({len(qs)} questions, {len(docs)} PDFs)",
        "model": llm.model,
        "ablations": {
            "1. Chunk-BM25 (flat 1000ch)": retrieval_metrics(p_chunk, golds),
            "2. PageCard-RRF (zero-LLM)": retrieval_metrics(p_card, golds),
            "3. w/o AnchorMap (C3 + B42)": _run_cfg("3. w/o AnchorMap (C3 + B42)", use_anchor=False, use_fu=True),
            "4. w/o Gap-Followup (AnchorMap+G7+C3)": _run_cfg("4. w/o Gap-Followup (AnchorMap+G7+C3)", use_anchor=True, use_fu=False),
            "5. AnchorDoc-Full (AnchorMap+G7+C3+B42)": _run_cfg("5. AnchorDoc-Full (AnchorMap+G7+C3+B42)", use_anchor=True, use_fu=True),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = ROOT / "results/financebench_150_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} ({results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
