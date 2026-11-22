"""T7 Credibility & Prefix-Cache Audit: 3-Run Live Variance + Offline Replay + Real cached_tokens.

Evaluates 30 stratified FinanceBench questions grouped by multi-question SEC 10-K PDFs on official
DeepSeek-V4.1-Flash (`temperature=0`, `--route auto`) across:
  - 3 independent fresh live API runs (measuring real MoE variance: Mean +/- Sample SD)
  - 3 offline SHA-cached replay runs (verifying exact 0.00% replay variance)
  - Real provider KV prefix-cache `cached_tokens` accounting on sequential multi-question PDFs
    with AnchorDoc v0.8.0's prefix-stable `anchordoc_nav` prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.finance_10k import Finance10KAdapter
from anchordoc.docbase import PdfDocument
from anchordoc.runtime import CachedLLM
from benchmarks.run_financebench_full import judge_answer, resolve_gold_pages, retrieval_metrics


def _resolve_api_key() -> str:
    key = os.environ.get("ANCHORDOC_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    key_file = os.environ.get("ANCHORDOC_LLM_KEY_FILE") or ""
    if not key and key_file:
        key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Set ANCHORDOC_LLM_KEY or ANCHORDOC_LLM_KEY_FILE.")
    return key


def _select_multi_question_30(samples_path: Path, pdf_dir: Path, target_n: int = 30) -> list[dict[str, Any]]:
    """Select 30 questions from multi-question PDFs (>=2 questions/doc), grouped by doc_name."""
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        doc_name = str(r.get("doc_name") or "")
        if (pdf_dir / f"{doc_name}.pdf").is_file():
            by_doc[doc_name].append(r)

    # Sort multi-question docs deterministically (most questions first, then doc_name)
    multi_docs = sorted(
        [(dn, qs) for dn, qs in by_doc.items() if len(qs) >= 2],
        key=lambda x: (-len(x[1]), x[0]),
    )
    selected: list[dict[str, Any]] = []
    for dn, qs in multi_docs:
        if len(selected) >= target_n:
            break
        remaining = target_n - len(selected)
        selected.extend(qs[:remaining])
    return selected


def _run_once(
    selected: list[dict[str, Any]],
    pdf_dir: Path,
    docs_cache: dict[str, PdfDocument],
    cache_dir: Path,
    judge_llm: CachedLLM,
    base_url: str,
    model: str,
    api_key: str,
    *,
    fresh: bool,
    route: str = "auto",
) -> dict[str, Any]:
    if fresh and cache_dir.exists():
        shutil.rmtree(cache_dir)
    llm = CachedLLM(cache_dir=cache_dir, model=model, base_url=base_url, api_key=api_key)
    t0 = time.perf_counter()

    # Group sequentially by doc_name so consecutive questions on the same PDF hit prefix KV cache
    adapters: dict[str, Finance10KAdapter] = {}
    preds_pages: list[list[int]] = []
    golds_pages: list[list[int]] = []
    judges: list[dict[str, Any]] = []
    quote_ok = 0
    ops_total = 0
    ops_grounded = 0

    for idx, row in enumerate(selected, start=1):
        dn = row["doc_name"]
        doc = docs_cache[dn]
        if dn not in adapters:
            adapters[dn] = Finance10KAdapter(doc, llm)
        adapter = adapters[dn]
        v = adapter.query(row["question"], item_id=f"T7-{idx:02d}", top_k=5, route=route)  # type: ignore[arg-type]
        tr_pages = list((v.retrieval_trace or {}).get("pages") or [])
        preds_pages.append(tr_pages)
        golds_pages.append(resolve_gold_pages(doc, row.get("evidence") or []))
        if v.quote_verified:
            quote_ok += 1
        meta = v.meta or {}
        if meta.get("calc_value") is not None:
            ops_total += 1
            if meta.get("operands_grounded"):
                ops_grounded += 1
        j = judge_answer(row["question"], row["answer"], v.answer, v.status == "answered", judge_llm)
        judges.append(j)

    elapsed = time.perf_counter() - t0
    st = llm.snapshot_stats()
    ret = retrieval_metrics(preds_pages, golds_pages)
    n = len(selected)
    corr = sum(1 for j in judges if j["verdict"] == "correct")
    inc = sum(1 for j in judges if j["verdict"] == "incorrect")
    abst = sum(1 for j in judges if j["verdict"] == "abstain")
    pt = int(st.get("prompt_tokens", 0) or 0)
    cached_pt = int(st.get("cached_tokens", 0) or 0)
    ct = int(st.get("completion_tokens", 0) or 0)
    calls = int(st.get("calls", 0) or 0)
    real_calls = int(st.get("real_calls", 0) or 0)
    cache_hits = int(st.get("cache_hits", 0) or 0)

    return {
        "n": n,
        "e2e_acc_%": round(corr / n * 100, 2),
        "incorrect_%": round(inc / n * 100, 2),
        "abstain_%": round(abst / n * 100, 2),
        "hit@5_%": ret["hit@5"],
        "all_gold@5_%": ret["all_gold@5"],
        "quote_verified_%": round(quote_ok / n * 100, 2),
        "operands_grounded": f"{ops_grounded}/{ops_total}",
        "avg_prompt_tokens": round(pt / n, 1),
        "avg_cached_tokens": round(cached_pt / n, 1),
        "prefix_cache_hit_rate_%": round(cached_pt / pt * 100, 2) if pt else 0.0,
        "total_prompt_tokens": pt,
        "total_cached_tokens": cached_pt,
        "total_completion_tokens": ct,
        "avg_calls": round(calls / n, 2),
        "real_api_calls": real_calls,
        "local_cache_hits": cache_hits,
        "elapsed_sec": round(elapsed, 2),
    }


def _mean_sd(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": round(mean, 2), "sd": round(sd, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description="T7 3-Run Live Variance + Prefix Cache Benchmark.")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--route", default="auto")
    ap.add_argument("--out", default="results/variance_t7_v08.json")
    args = ap.parse_args()

    base_url = os.environ.get("ANCHORDOC_LLM_BASE", "https://api.deepseek.com/v1")
    model = os.environ.get("ANCHORDOC_LLM_MODEL", "deepseek-flash")
    api_key = _resolve_api_key()

    pdf_dir = ROOT / "data" / "financebench_pdfs"
    samples_path = ROOT / "data" / "financebench_merged.jsonl"
    selected = _select_multi_question_30(samples_path, pdf_dir, target_n=args.n)
    doc_names = sorted({r["doc_name"] for r in selected})
    docs_cache = {dn: PdfDocument.from_pdf(pdf_dir / f"{dn}.pdf") for dn in doc_names}
    print(f"T7 Variance Audit: {len(selected)} questions across {len(doc_names)} multi-question SEC PDFs | model={model}")

    judge_llm = CachedLLM(
        cache_dir=ROOT / "cache" / "var_t7_judge_v08",
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    live_runs: list[dict[str, Any]] = []
    for r in range(1, args.runs + 1):
        cdir = ROOT / "cache" / f"var_t7_live_run{r}"
        res = _run_once(
            selected, pdf_dir, docs_cache, cdir, judge_llm, base_url, model, api_key,
            fresh=True, route=args.route,
        )
        live_runs.append(res)
        print(
            f"  [Live Run {r}/{args.runs}] E2E={res['e2e_acc_%']:.2f}% | Hit@5={res['hit@5_%']:.1f}% | "
            f"AllGold@5={res['all_gold@5_%']:.1f}% | QVer={res['quote_verified_%']:.1f}% | "
            f"PromptTok={res['avg_prompt_tokens']:.0f} | PrefixCached={res['avg_cached_tokens']:.0f} "
            f"({res['prefix_cache_hit_rate_%']:.1f}%) | {res['elapsed_sec']}s"
        )

    # 3 offline replay runs of run 1 cache (must have 0 real_api_calls and SD=0.00%)
    replay_runs: list[dict[str, Any]] = []
    replay_cache = ROOT / "cache" / "var_t7_live_run1"
    for r in range(1, args.runs + 1):
        res = _run_once(
            selected, pdf_dir, docs_cache, replay_cache, judge_llm, base_url, model, api_key,
            fresh=False, route=args.route,
        )
        replay_runs.append(res)

    summary = {
        "benchmark": f"FinanceBench T7 Variance & Prefix-Cache Audit ({len(selected)}q across {len(doc_names)} multi-question PDFs)",
        "model": model,
        "base_url": base_url,
        "route": args.route,
        "docs": doc_names,
        "live_runs": live_runs,
        "live_summary": {
            "e2e_acc_%": _mean_sd([r["e2e_acc_%"] for r in live_runs]),
            "hit@5_%": _mean_sd([r["hit@5_%"] for r in live_runs]),
            "all_gold@5_%": _mean_sd([r["all_gold@5_%"] for r in live_runs]),
            "quote_verified_%": _mean_sd([r["quote_verified_%"] for r in live_runs]),
            "avg_prompt_tokens": _mean_sd([r["avg_prompt_tokens"] for r in live_runs]),
            "avg_cached_tokens": _mean_sd([r["avg_cached_tokens"] for r in live_runs]),
            "prefix_cache_hit_rate_%": _mean_sd([r["prefix_cache_hit_rate_%"] for r in live_runs]),
            "avg_calls": _mean_sd([r["avg_calls"] for r in live_runs]),
        },
        "offline_replay_runs": replay_runs,
        "offline_replay_summary": {
            "e2e_acc_%": _mean_sd([r["e2e_acc_%"] for r in replay_runs]),
            "hit@5_%": _mean_sd([r["hit@5_%"] for r in replay_runs]),
            "real_api_calls_sum": sum(r["real_api_calls"] for r in replay_runs),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ls = summary["live_summary"]
    rs = summary["offline_replay_summary"]
    print("\n=== T7 Variance & Prefix-Cache Summary ===")
    print(f"  Live 3-Run E2E Acc   : {ls['e2e_acc_%']['mean']:.2f}% ± {ls['e2e_acc_%']['sd']:.2f}%")
    print(f"  Live 3-Run Hit@5     : {ls['hit@5_%']['mean']:.2f}% ± {ls['hit@5_%']['sd']:.2f}%")
    print(f"  Live 3-Run QVerified : {ls['quote_verified_%']['mean']:.2f}% ± {ls['quote_verified_%']['sd']:.2f}%")
    print(f"  Live 3-Run PromptTok : {ls['avg_prompt_tokens']['mean']:.0f} ± {ls['avg_prompt_tokens']['sd']:.0f}")
    print(f"  Live 3-Run PrefixHit : {ls['prefix_cache_hit_rate_%']['mean']:.1f}% ± {ls['prefix_cache_hit_rate_%']['sd']:.1f}% ({ls['avg_cached_tokens']['mean']:.0f} cached tok/q)")
    print(f"  Offline Replay (3x)  : {rs['e2e_acc_%']['mean']:.2f}% ± {rs['e2e_acc_%']['sd']:.2f}% (real_api_calls={rs['real_api_calls_sum']})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
