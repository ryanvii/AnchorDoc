"""Head-to-Head Competitive Comparison on FinanceBench:
  Part A (Full 150/150, 84 PDFs, 12,013 pages):
    - Chunk-BM25
    - Dense Vector RAG (BAAI/bge-small-en-v1.5 ONNX embeddings)
    - Hybrid Vector + BM25 (RRF)
    - AnchorDoc v0.4 (Zero-LLM AnchorMap + G7 + C3 + B42)

  Part B (8 real 75-503 page SEC 10-Ks, 1,814 pages, 39 official questions):
    - LLM-Tree-Index (PageIndex-style upfront LLM hierarchical tree construction + tree search)
    - AnchorDoc v0.4 (Zero-LLM AnchorMap, 0 indexing LLM calls)
    Measures EXACT indexing wall time, indexing LLM calls, indexing tokens, Hit@5, AllGold@5, and E2E Acc!
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.runtime import CachedLLM, rrf, tokenize
from benchmarks.run_financebench_full import judge_answer, resolve_gold_pages, retrieval_metrics, summarize_qa


TREE_COMPARE_DOCS = [
    "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K",
    "BOEING_2022_10K",
    "PEPSICO_2022_10K",
    "AMCOR_2023_10K",
    "3M_2022_10K",
    "AES_2022_10K",
    "BESTBUY_2023_10K",
]


def _embed_matrix(embedder: TextEmbedding, texts: list[str]) -> np.ndarray:
    vecs = list(embedder.embed(texts, batch_size=64))
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def build_llm_tree_for_pdf(doc: PdfDocument, llm: CachedLLM, window_pages: int = 12) -> list[dict[str, Any]]:
    """PageIndex-style upfront LLM hierarchical tree indexing over an entire PDF.

    Scans consecutive page windows with the LLM to construct section nodes
    (title, start_page, end_page, summary) for every part of the document.
    """
    windows: list[tuple[int, int]] = []
    for s in range(1, doc.num_pages + 1, window_pages):
        e = min(doc.num_pages, s + window_pages - 1)
        windows.append((s, e))

    def _summarize_window(rng: tuple[int, int]) -> list[dict[str, Any]]:
        s, e = rng
        # Pack compact excerpt of every page in this window
        body = "\n\n".join(
            f"[Page {p}]\n{doc.page_text(p)[:550]}" for p in range(s, e + 1)
        )
        parsed, _ = llm.chat_json(
            "comp_llmtree_index",
            "You are a document tree indexer (PageIndex style). Return JSON only.",
            (
                f"Pages {s} to {e} of a long financial PDF:\n{body}\n\n"
                "Identify the main structural sections / financial statements / footnotes in this page range.\n"
                'Return JSON: {"nodes": [{"title": "...", "start_page": int, "end_page": int, "summary": "1-sentence summary"}]}'
            ),
            max_tokens=450,
        )
        nodes = parsed.get("nodes") if isinstance(parsed.get("nodes"), list) else []
        clean = []
        for nd in nodes:
            if not isinstance(nd, dict):
                continue
            sp = int(nd.get("start_page") or s) if isinstance(nd.get("start_page"), (int, float)) else s
            ep = int(nd.get("end_page") or e) if isinstance(nd.get("end_page"), (int, float)) else e
            sp = max(s, min(e, sp))
            ep = max(sp, min(e, ep))
            clean.append(
                {
                    "title": str(nd.get("title") or f"Pages {sp}-{ep}")[:90],
                    "start_page": sp,
                    "end_page": ep,
                    "summary": str(nd.get("summary") or "")[:160],
                }
            )
        return clean or [{"title": f"Pages {s}-{e}", "start_page": s, "end_page": e, "summary": ""}]

    tree: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for batch in ex.map(_summarize_window, windows):
            tree.extend(batch)
    tree.sort(key=lambda x: (x["start_page"], x["end_page"]))
    return tree


def search_llm_tree(doc: PdfDocument, tree: list[dict[str, Any]], question: str, llm: CachedLLM, top_k: int = 5) -> list[int]:
    """PageIndex-style reasoning search over the LLM-constructed tree."""
    tree_txt = "\n".join(
        f"- [p.{n['start_page']}-{n['end_page']}] {n['title']}: {n['summary']}"
        for n in tree[:120]
    )
    parsed, _ = llm.chat_json(
        "comp_llmtree_search",
        "You are a tree-reasoning document navigator. Return JSON only.",
        (
            f"Question: {question}\n\n"
            f"Document Tree Index:\n{tree_txt}\n\n"
            "Reason over the tree and select up to 5 specific physical pages most likely to directly answer the question, "
            "most relevant FIRST.\n"
            'Return JSON: {"pages": [int], "reasoning": ""}'
        ),
        max_tokens=220,
    )
    pages = [
        int(p) for p in (parsed.get("pages") or [])
        if isinstance(p, (int, float)) and 1 <= int(p) <= doc.num_pages
    ]
    # Deduplicate preserving order
    out: list[int] = []
    for p in pages:
        if p not in out:
            out.append(p)
        if len(out) >= top_k:
            break
    return out


def answer_dense_rag_e2e(chunks_top5: list[tuple[int, str]], question: str, llm: CachedLLM) -> dict[str, Any]:
    ctx = "\n\n---\n\n".join(f"[Page {p}]\n{txt}" for p, txt in chunks_top5)
    parsed, _ = llm.chat_json(
        "comp_dense_e2e",
        "You are a financial analyst answering questions strictly from retrieved SEC excerpts. Return JSON only.",
        (
            f"Question: {question}\n\nRetrieved Excerpts:\n{ctx}\n\n"
            "Answer concisely based ONLY on the excerpts. Set can_answer=false if insufficient.\n"
            'Return JSON: {"can_answer": true/false, "answer": "", "cited_pages": [int]}'
        ),
        max_tokens=600,
    )
    return {
        "can_answer": bool(parsed.get("can_answer", True)),
        "answer": str(parsed.get("answer") or "").strip(),
    }


def main() -> None:
    t0 = time.time()
    fb = [json.loads(ln) for ln in open(ROOT / "data/financebench_merged.jsonl", encoding="utf-8")]
    pdf_dir = ROOT / "data/financebench_pdfs"
    qs = [r for r in fb if (pdf_dir / f"{r['doc_name']}.pdf").is_file()]
    unique_docs = sorted({q["doc_name"] for q in qs})

    print(f"=== Part A: Real Dense Vector RAG (BGE ONNX) vs AnchorDoc on ALL {len(qs)} FinanceBench Questions ===")
    t_load0 = time.time()
    docs = {d: PdfDocument.from_pdf(pdf_dir / f"{d}.pdf", backend="pypdfium2") for d in unique_docs}
    indices = {d: DocumentIndex(docs[d]) for d in unique_docs}
    zero_llm_idx_sec = time.time() - t_load0
    tot_pages = sum(d.num_pages for d in docs.values())
    golds = [resolve_gold_pages(docs[q["doc_name"]], q["evidence"]) for q in qs]

    # Embed all chunks + pages per doc with BAAI/bge-small-en-v1.5 (CPU ONNX)
    embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    t_emb0 = time.time()
    q_vecs = _embed_matrix(embedder, [q["question"] for q in qs])

    chunk_embs: dict[str, np.ndarray] = {}
    page_embs: dict[str, np.ndarray] = {}
    for d in unique_docs:
        ch_texts = [c[1] for c in indices[d].chunks] or [""]
        chunk_embs[d] = _embed_matrix(embedder, ch_texts)
        pg_texts = [(p[:1800] if p.strip() else "blank") for p in docs[d].pages]
        page_embs[d] = _embed_matrix(embedder, pg_texts)
    dense_idx_sec = time.time() - t_emb0
    print(f"Dense BGE ONNX indexed {len(docs)} PDFs ({tot_pages} pages) in {dense_idx_sec:.1f}s ({dense_idx_sec/len(docs):.2f}s/PDF)")

    p_dense_chunk: list[list[int]] = []
    p_dense_page: list[list[int]] = []
    p_hybrid_rrf: list[list[int]] = []
    dense_top5_chunks: list[list[tuple[int, str]]] = []

    for i, q in enumerate(qs):
        d = q["doc_name"]
        qv = q_vecs[i]
        # Dense chunk retrieval
        cs = chunk_embs[d] @ qv
        c_ord = np.argsort(-cs)
        dc_pages: list[int] = []
        top5_ch: list[tuple[int, str]] = []
        for ci in c_ord[:5]:
            top5_ch.append(indices[d].chunks[int(ci)])
        dense_top5_chunks.append(top5_ch)
        for ci in c_ord:
            pg = indices[d].chunks[int(ci)][0]
            if pg not in dc_pages:
                dc_pages.append(pg)
                if len(dc_pages) >= 10:
                    break
        p_dense_chunk.append(dc_pages)

        # Dense page retrieval
        ps = page_embs[d] @ qv
        dp_pages = [int(pi) + 1 for pi in np.argsort(-ps)[:10]]
        p_dense_page.append(dp_pages)

        # Hybrid Dense Page + BM25 PageCard RRF
        bm_pages = indices[d].rank_pagecard_rrf(q["question"], top_k=20)
        hyb = rrf(dp_pages, bm_pages, k=60)[:10]
        p_hybrid_rrf.append(hyb)

    # Load AnchorDoc's already-verified 150-question results for direct comparison
    ad150 = json.loads((ROOT / "results/financebench_150.json").read_text(encoding="utf-8"))
    ad_by_id = {it["id"]: it for it in ad150["per_item"]}
    p_anchor = [ad_by_id[q["financebench_id"]]["anchor_pages"] for q in qs]

    llm = CachedLLM(cache_dir=ROOT / "cache/competitive_comp")

    # End-to-End QA for Dense-Chunk-BGE
    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=8) as ex:
        dense_ans = list(ex.map(lambda qc: answer_dense_rag_e2e(qc[1], qc[0]["question"], llm), zip(qs, dense_top5_chunks)))
    dense_e2e_stats = llm.snapshot_stats()

    with ThreadPoolExecutor(max_workers=8) as ex:
        j_dense = list(ex.map(lambda qa: judge_answer(qa[0]["question"], qa[0]["answer"], qa[1]["answer"], qa[1]["can_answer"], llm), zip(qs, dense_ans)))

    part_a = {
        "n_questions": len(qs),
        "n_pdfs": len(docs),
        "total_pages": tot_pages,
        "indexing_wall_sec": {
            "AnchorDoc (Zero-LLM)": round(zero_llm_idx_sec, 1),
            "Dense-BGE-ONNX (CPU)": round(dense_idx_sec, 1),
        },
        "retrieval": {
            "Chunk-BM25": ad150["retrieval"]["Chunk-BM25 (flat 1000ch)"],
            "Dense-Chunk-BGE (bge-small-en-v1.5)": retrieval_metrics(p_dense_chunk, golds),
            "Dense-Page-BGE (bge-small-en-v1.5)": retrieval_metrics(p_dense_page, golds),
            "Hybrid-BGE+BM25 (RRF)": retrieval_metrics(p_hybrid_rrf, golds),
            "AnchorDoc v0.4 (Zero-LLM AnchorMap+G7+C3+B42)": ad150["retrieval"]["AnchorDoc-Full (AnchorMap+G7+C3+B42)"],
        },
        "e2e_qa": {
            "ChunkRAG-BM25-E2E": ad150["e2e_qa"]["ChunkRAG-E2E"],
            "DenseRAG-BGE-E2E": summarize_qa(j_dense, dense_e2e_stats),
            "AnchorDoc-v0.4-E2E": ad150["e2e_qa"]["AnchorDoc-E2E"],
        },
    }

    print("\n=== Part A Results (ALL 150 FinanceBench Questions, 12,013 Pages) ===")
    for name, m in part_a["retrieval"].items():
        print(f"  RET | {name:46s} | Hit@1={m['hit@1']:5.1f}% | Hit@5={m['hit@5']:5.1f}% | AllGold@5={m['all_gold@5']:5.1f}% | MRR={m['mrr@10']:.4f}")
    for name, m in part_a["e2e_qa"].items():
        print(f"  E2E | {name:46s} | Acc={m['e2e_acc_%']:5.1f}% | Wrong={m['incorrect_%']:5.1f}% | Abstain={m['abstain_%']:5.1f}% | AnsPrec={m['answered_precision_%']:5.1f}%")

    # =========================================================================
    # Part B: LLM-Tree-Index (PageIndex-style) vs AnchorDoc on 8 SEC 10-Ks (1,814 pages, 39q)
    # =========================================================================
    print("\n=== Part B: LLM-Tree-Index (PageIndex-style) vs AnchorDoc on 8 SEC 10-Ks (1,814 pages, 39q) ===")
    sub_qs = [q for q in qs if q["doc_name"] in TREE_COMPARE_DOCS]
    sub_golds = [resolve_gold_pages(docs[q["doc_name"]], q["evidence"]) for q in sub_qs]
    sub_pages = sum(docs[d].num_pages for d in TREE_COMPARE_DOCS)

    # Measure upfront LLM tree indexing cost & wall time
    llm.reset_stats()
    t_tree0 = time.time()
    trees: dict[str, list[dict]] = {}
    for d in TREE_COMPARE_DOCS:
        trees[d] = build_llm_tree_for_pdf(docs[d], llm, window_pages=12)
    tree_idx_sec = time.time() - t_tree0
    tree_idx_stats = llm.snapshot_stats()

    # Measure tree search + E2E QA (using exact same FactAdjudicator!)
    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=6) as ex:
        p_tree = list(ex.map(lambda q: search_llm_tree(docs[q["doc_name"]], trees[q["doc_name"]], q["question"], llm, top_k=5), sub_qs))
    tree_search_stats = llm.snapshot_stats()

    adj = FactAdjudicator(llm=llm)
    llm.reset_stats()
    with ThreadPoolExecutor(max_workers=6) as ex:
        tree_verdicts = list(ex.map(lambda qp: adj.answer_numerical(docs[qp[0]["doc_name"]], qp[0]["question"], qp[1], item_id=qp[0]["financebench_id"]), zip(sub_qs, p_tree)))
    tree_qa_stats = llm.snapshot_stats()

    with ThreadPoolExecutor(max_workers=6) as ex:
        j_tree = list(ex.map(lambda qv: judge_answer(qv[0]["question"], qv[0]["answer"], qv[1].answer, qv[1].status == "answered", llm), zip(sub_qs, tree_verdicts)))

    # AnchorDoc stats on exact same 39 questions
    sub_ad_pages = [ad_by_id[q["financebench_id"]]["anchor_pages"] for q in sub_qs]
    sub_ad_correct = sum(1 for q in sub_qs if ad_by_id[q["financebench_id"]]["anchor_verdict"] == "correct")
    sub_ad_acc = round(sub_ad_correct / len(sub_qs) * 100, 1)

    t_ad0 = time.time()
    for d in TREE_COMPARE_DOCS:
        DocumentIndex(docs[d]).build_structural_outline()
    ad_sub_idx_sec = time.time() - t_ad0

    part_b = {
        "docs": TREE_COMPARE_DOCS,
        "n_pdfs": len(TREE_COMPARE_DOCS),
        "total_pages": sub_pages,
        "n_questions": len(sub_qs),
        "LLM-Tree-Index (PageIndex-style)": {
            "indexing_wall_sec": round(tree_idx_sec, 1),
            "indexing_sec_per_pdf": round(tree_idx_sec / len(TREE_COMPARE_DOCS), 2),
            "indexing_llm_calls": tree_idx_stats["calls"],
            "indexing_calls_per_pdf": round(tree_idx_stats["calls"] / len(TREE_COMPARE_DOCS), 1),
            "indexing_prompt_tokens": tree_idx_stats["prompt_tokens"],
            "indexing_tokens_per_pdf": round(tree_idx_stats["prompt_tokens"] / len(TREE_COMPARE_DOCS), 0),
            "retrieval": retrieval_metrics(p_tree, sub_golds),
            "e2e_qa": summarize_qa(j_tree, {
                "calls": tree_search_stats["calls"] + tree_qa_stats["calls"],
                "prompt_tokens": tree_search_stats["prompt_tokens"] + tree_qa_stats["prompt_tokens"],
            }),
        },
        "AnchorDoc v0.4 (Zero-LLM AnchorMap)": {
            "indexing_wall_sec": round(ad_sub_idx_sec, 2),
            "indexing_sec_per_pdf": round(ad_sub_idx_sec / len(TREE_COMPARE_DOCS), 3),
            "indexing_llm_calls": 0,
            "indexing_calls_per_pdf": 0.0,
            "indexing_prompt_tokens": 0,
            "indexing_tokens_per_pdf": 0,
            "retrieval": retrieval_metrics(sub_ad_pages, sub_golds),
            "e2e_acc_%": sub_ad_acc,
        },
    }

    lt = part_b["LLM-Tree-Index (PageIndex-style)"]
    ad = part_b["AnchorDoc v0.4 (Zero-LLM AnchorMap)"]
    print(f"  LLM-Tree-Index | Idx: {lt['indexing_sec_per_pdf']}s/PDF, {lt['indexing_calls_per_pdf']} calls/PDF, {int(lt['indexing_tokens_per_pdf'])} tok/PDF | Hit@5={lt['retrieval']['hit@5']}% AllGold@5={lt['retrieval']['all_gold@5']}% | E2E={lt['e2e_qa']['e2e_acc_%']}%")
    print(f"  AnchorDoc v0.4 | Idx: {ad['indexing_sec_per_pdf']}s/PDF, 0 calls/PDF, 0 tok/PDF | Hit@5={ad['retrieval']['hit@5']}% AllGold@5={ad['retrieval']['all_gold@5']}% | E2E={ad['e2e_acc_%']}%")

    out = {"model": llm.model, "part_a_dense_150": part_a, "part_b_llmtree_39": part_b, "elapsed_sec": round(time.time() - t0, 1)}
    out_path = ROOT / "results/competitive_comparison.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path} ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
