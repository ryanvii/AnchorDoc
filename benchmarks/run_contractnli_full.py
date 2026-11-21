"""ContractNLI-Long Benchmark Runner for AnchorDoc (N=150 full contracts >= 10k chars).

Compares on the exact same LLM (temperature=0):
  1. ChunkRAG-Direct     (flat 800-char chunks + direct verdict)
  2. LongContext-Direct  (full 10k-42k contract stuffed into LLM)
  3. AnchorDoc (Ours)    (SpanCard + Dual BM25-RRF + B42 Two-Step Gap-Followup + QuoteGate + Confidence Selective Prediction)

Computes Overall Accuracy, Macro-F1, False-Pass Rate, Risk-Coverage (Selective Prediction),
and exact Token Cost per contract.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.legal_nda import LegalNDAAdapter
from anchordoc.runtime import BM25Okapi, CachedLLM, tokenize, verify_quote

VALID_LABELS = ("entailment", "contradiction", "neutral")


def _norm_label(val: Any) -> str:
    s = str(val or "").strip().lower()
    if "entail" in s or s == "pass":
        return "entailment"
    if "contra" in s or s == "fail":
        return "contradiction"
    return "neutral"


def build_sample(data_path: Path, per_class: int = 50, min_chars: int = 10000, seed: int = 42) -> list[dict]:
    rows = [json.loads(ln) for ln in open(data_path, encoding="utf-8")]
    test_docs = [r for r in rows if r.get("subset") == "test" and len(r.get("premise") or "") >= min_chars]
    buckets: dict[str, list[dict]] = {lbl: [] for lbl in VALID_LABELS}
    for doc_idx, doc in enumerate(test_docs):
        premise = doc["premise"]
        for h_idx, item in enumerate(doc["hypothesises/labels"]):
            lbl = item["label"].strip().lower()
            if lbl in buckets:
                buckets[lbl].append(
                    {
                        "id": f"cnli_t{doc_idx:03d}_h{h_idx:02d}",
                        "doc_idx": doc_idx,
                        "premise_len": len(premise),
                        "premise": premise,
                        "hypothesis": item["hypothesis"],
                        "gold": lbl,
                    }
                )
    rng = random.Random(seed)
    out: list[dict] = []
    for lbl in VALID_LABELS:
        pool = list(buckets[lbl])
        rng.shuffle(pool)
        out.extend(pool[:per_class])
    rng.shuffle(out)
    return out


def run_chunk_baseline(premise: str, hypothesis: str, llm: CachedLLM) -> dict[str, Any]:
    chunks = [premise[s : s + 800].strip() for s in range(0, len(premise), 680) if premise[s : s + 800].strip()]
    bm = BM25Okapi([tokenize(c) for c in chunks])
    scores = bm.get_scores(tokenize(hypothesis))
    top4 = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:4]]
    ctx = "\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(top4))
    parsed, _ = llm.chat_json(
        "cnli150_chunk",
        "You are a legal contract compliance reviewer. Return JSON only.",
        (
            f"Requirement:\n{hypothesis}\n\nChunks:\n{ctx}\n\n"
            "Classify as entailment, contradiction, or neutral, with confidence ('high'|'medium'|'low') and verbatim quote.\n"
            'Return JSON: {"label": "entailment|contradiction|neutral", "confidence": "high|medium|low", "quote": ""}'
        ),
        max_tokens=240,
    )
    lbl = _norm_label(parsed.get("label"))
    q = str(parsed.get("quote") or "").strip()
    qv, _ = verify_quote(q, premise) if q else (False, 0.0)
    return {"label": lbl, "confidence": str(parsed.get("confidence") or "medium").lower(), "quote_verified": qv}


def run_long_context_baseline(premise: str, hypothesis: str, llm: CachedLLM) -> dict[str, Any]:
    parsed, _ = llm.chat_json(
        "cnli150_longctx",
        "You are a legal contract compliance reviewer. Return JSON only.",
        (
            f"Requirement:\n{hypothesis}\n\nFull Contract:\n{premise[:55000]}\n\n"
            "Classify as entailment, contradiction, or neutral, with confidence ('high'|'medium'|'low') and verbatim quote.\n"
            'Return JSON: {"label": "entailment|contradiction|neutral", "confidence": "high|medium|low", "quote": ""}'
        ),
        max_tokens=240,
    )
    lbl = _norm_label(parsed.get("label"))
    q = str(parsed.get("quote") or "").strip()
    qv, _ = verify_quote(q, premise) if q else (False, 0.0)
    return {"label": lbl, "confidence": str(parsed.get("confidence") or "medium").lower(), "quote_verified": qv}


def run_anchordoc(premise: str, hypothesis: str, item_id: str, llm: CachedLLM) -> dict[str, Any]:
    adapter = LegalNDAAdapter(premise, llm)
    v = adapter.check_clause(hypothesis, item_id=item_id)
    lbl = _norm_label(v.answer)
    # QuoteGate enforcement on pass/fail
    if lbl in {"entailment", "contradiction"} and not v.quote_verified:
        lbl = "neutral"
    conf = str((v.facts or {}).get("confidence") or "medium").lower()
    return {"label": lbl, "confidence": conf, "quote_verified": v.quote_verified}


def score_preds(preds: list[dict], golds: list[str], stats: dict) -> dict[str, Any]:
    n = len(golds)
    acc = sum(1 for p, g in zip(preds, golds) if p["label"] == g) / n * 100
    f1s = []
    per_cls = {}
    for lbl in VALID_LABELS:
        tp = sum(1 for p, g in zip(preds, golds) if p["label"] == lbl and g == lbl)
        fp = sum(1 for p, g in zip(preds, golds) if p["label"] == lbl and g != lbl)
        fn = sum(1 for p, g in zip(preds, golds) if p["label"] != lbl and g == lbl)
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        f1s.append(f1)
        per_cls[lbl] = {"p": round(pr * 100, 1), "r": round(rc * 100, 1), "f1": round(f1 * 100, 1)}

    non_ent = [(p, g) for p, g in zip(preds, golds) if g != "entailment"]
    fer = sum(1 for p, _ in non_ent if p["label"] == "entailment") / len(non_ent) * 100
    c2e = sum(1 for p, g in zip(preds, golds) if g == "contradiction" and p["label"] == "entailment")

    # Selective prediction: high-confidence auto-decided subset
    hi = [(p, g) for p, g in zip(preds, golds) if p["confidence"] == "high"]
    hi_cov = len(hi) / n * 100
    hi_acc = (sum(1 for p, g in hi if p["label"] == g) / len(hi) * 100) if hi else 0.0
    hi_ne = [(p, g) for p, g in hi if g != "entailment"]
    hi_fer = (sum(1 for p, _ in hi_ne if p["label"] == "entailment") / len(hi_ne) * 100) if hi_ne else 0.0

    return {
        "n": n,
        "accuracy_%": round(acc, 1),
        "macro_f1_%": round(sum(f1s) / 3 * 100, 1),
        "false_pass_rate_%": round(fer, 1),
        "contra_to_ent": c2e,
        "high_conf_coverage_%": round(hi_cov, 1),
        "high_conf_acc_%": round(hi_acc, 1),
        "high_conf_fer_%": round(hi_fer, 1),
        "avg_prompt_tokens": round(stats["prompt_tokens"] / n, 0),
        "per_class": per_cls,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/contract_nli_long.jsonl"))
    ap.add_argument("--cache-dir", default=str(ROOT / "cache/contractnli_full"))
    ap.add_argument("--out", default=str(ROOT / "results/contractnli_150.json"))
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    sample = build_sample(Path(args.data), per_class=args.per_class, min_chars=10000, seed=42)
    golds = [s["gold"] for s in sample]
    print(f"ContractNLI-Long Run: N={len(sample)} ({dict(Counter(golds))})")

    llm = CachedLLM(cache_dir=args.cache_dir)
    runners = [
        ("ChunkRAG-Direct", lambda s: run_chunk_baseline(s["premise"], s["hypothesis"], llm)),
        ("LongContext-Direct", lambda s: run_long_context_baseline(s["premise"], s["hypothesis"], llm)),
        ("AnchorDoc (Ours)", lambda s: run_anchordoc(s["premise"], s["hypothesis"], s["id"], llm)),
    ]

    summary: dict[str, Any] = {"model": llm.model, "n": len(sample), "methods": {}}
    for name, fn in runners:
        llm.reset_stats()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            preds = list(ex.map(fn, sample))
        m = score_preds(preds, golds, llm.snapshot_stats())
        summary["methods"][name] = m
        print(
            f"  {name:20s} | Acc={m['accuracy_%']:5.1f}% | MacroF1={m['macro_f1_%']:5.1f}% | "
            f"FER={m['false_pass_rate_%']:5.1f}% (C->E:{m['contra_to_ent']}) | "
            f"HiConfCov={m['high_conf_coverage_%']:5.1f}% (HiAcc={m['high_conf_acc_%']:5.1f}%, HiFER={m['high_conf_fer_%']:4.1f}%) | "
            f"Tok={int(m['avg_prompt_tokens'])}"
        )

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path} ({summary['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
