"""Publication-grade statistical verification for AnchorDoc Technical Report (T1–T7).

Computes directly from saved JSON result files (zero fabricated numbers):
- Bootstrap 95% confidence intervals (10,000 resamples, fixed seed=42)
- Exact two-sided paired McNemar p-values between AnchorDoc and baselines
- T6 Verifiability & CiteOrDie audit metrics (quote_verified %, operand_grounded %,
  composition breakdown, MM honest grounding tier accuracy)
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_ci(binary: list[int], n_boot: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean_%, ci_low_%, ci_high_%) via percentile bootstrap."""
    n = len(binary)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(binary[rng.randrange(n)] for _ in range(n))
        means.append(s / n * 100.0)
    means.sort()
    mean = sum(binary) / n * 100.0
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return round(mean, 1), round(lo, 1), round(hi, 1)


def exact_mcnemar_p(a: list[int], b: list[int]) -> dict[str, Any]:
    """Exact two-sided paired McNemar test between two binary outcome vectors (a vs b)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # b wins
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # a wins
    n = n01 + n10
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    k = min(n01, n10)
    # Two-sided exact binomial p-value under H0: p=0.5
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_val = min(1.0, 2.0 * tail)
    return {"a_only": n10, "b_only": n01, "p_value": p_val}


def analyze_financebench(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    items = d.get("per_item", [])
    ad_ok = [1 if r.get("anchor_verdict") == "correct" else 0 for r in items]
    ch_ok = [1 if r.get("chunk_verdict") == "correct" else 0 for r in items]
    qv = [1 if r.get("quote_verified") else 0 for r in items]
    calc_items = [r for r in items if r.get("calc_used")]
    og_list = [r for r in calc_items if isinstance(r.get("meta"), dict) and "operand_grounded" in r["meta"]]
    og_ok = sum(1 for r in og_list if r["meta"]["operand_grounded"])
    comp_counts = Counter((r.get("meta") or {}).get("composition", "N/A") for r in items)
    return {
        "file": path.name,
        "model": d.get("model"),
        "n": len(items),
        "anchor_e2e_ci": bootstrap_ci(ad_ok),
        "chunk_e2e_ci": bootstrap_ci(ch_ok),
        "mcnemar_anchor_vs_chunk": exact_mcnemar_p(ad_ok, ch_ok),
        "quote_verified_%": round(sum(qv) / max(1, len(qv)) * 100, 1),
        "calc_used_n": len(calc_items),
        "operand_grounded_n": f"{og_ok}/{len(og_list)}" if og_list else "N/A",
        "operand_grounded_%": round(og_ok / len(og_list) * 100, 1) if og_list else None,
        "composition_counts": dict(comp_counts),
        "retrieval": d.get("retrieval", {}).get("AnchorDoc-Full (AnchorMap+G7+C3+B42)"),
        "e2e_qa": d.get("e2e_qa", {}).get("AnchorDoc-E2E"),
    }


def analyze_mmlongbench(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    key = next((k for k in d.get("per_item", {}) if "AnchorDoc" in k), None)
    items = d.get("per_item", {}).get(key, []) if key else []
    overall = [1 if r.get("correct") else 0 for r in items]
    ans = [1 if r.get("correct") else 0 for r in items if not r.get("unanswerable")]
    charts = [1 if r.get("correct") else 0 for r in items if not r.get("unanswerable") and r.get("visual")]
    unans = [1 if r.get("correct") else 0 for r in items if r.get("unanswerable")]

    # T6: Accuracy stratified by honest grounding tier
    by_grounding: dict[str, list[int]] = {}
    for r in items:
        if r.get("status") == "answered" and isinstance(r.get("meta"), dict):
            g = r["meta"].get("grounding", "unknown")
            by_grounding.setdefault(g, []).append(1 if r.get("correct") else 0)
    grounding_acc = {
        g: {"n": len(v), "acc_%": round(sum(v) / len(v) * 100, 1)}
        for g, v in sorted(by_grounding.items())
    }
    return {
        "file": path.name,
        "model": d.get("model"),
        "n": len(items),
        "overall_ci": bootstrap_ci(overall),
        "answerable_ci": bootstrap_ci(ans),
        "chart_fig_ci": bootstrap_ci(charts),
        "unanswerable_ci": bootstrap_ci(unans),
        "grounding_tier_acc": grounding_acc,
        "methods": d.get("methods"),
        "retrieval": d.get("retrieval_on_answerable"),
    }


def analyze_contractnli(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"file": path.name, "model": d.get("model"), "methods": d.get("methods")}
    per_item = d.get("per_item", {})
    ad_key = next((k for k in per_item if "AnchorDoc" in k), None)
    lc_key = next((k for k in per_item if "LongContext" in k), None)
    if ad_key:
        ad_items = per_item[ad_key]
        ad_ok = [1 if r.get("gold") == r.get("label", r.get("pred")) else 0 for r in ad_items]
        qv = [1 if r.get("quote_verified") else 0 for r in ad_items]
        out["anchor_ci"] = bootstrap_ci(ad_ok)
        out["quote_verified_%"] = round(sum(qv) / max(1, len(qv)) * 100, 1)
        if lc_key:
            lc_items = per_item[lc_key]
            lc_ok = [1 if r.get("gold") == r.get("label", r.get("pred")) else 0 for r in lc_items]
            out["longctx_ci"] = bootstrap_ci(lc_ok)
            out["mcnemar_anchor_vs_longctx"] = exact_mcnemar_p(ad_ok, lc_ok)
    return out


def main() -> None:
    res_dir = ROOT / "results"
    report: dict[str, Any] = {"financebench": {}, "mmlongbench": {}, "contractnli": {}}

    for name in ("financebench_150.json", "financebench_150_v06.json", "financebench_150_v07_auto.json", "financebench_150_v071_v41flash.json"):
        p = res_dir / name
        if p.is_file():
            report["financebench"][name] = analyze_financebench(p)

    # Paired McNemar v0.6 vs v0.5 on FinanceBench 150
    p5, p6 = res_dir / "financebench_150.json", res_dir / "financebench_150_v06.json"
    if p5.is_file() and p6.is_file():
        a5 = [1 if r["anchor_verdict"] == "correct" else 0 for r in json.loads(p5.read_text())["per_item"]]
        a6 = [1 if r["anchor_verdict"] == "correct" else 0 for r in json.loads(p6.read_text())["per_item"]]
        report["financebench"]["mcnemar_v06_vs_v05"] = exact_mcnemar_p(a6, a5)

    for name in ("mmlongbench_101_v06.json", "mmlongbench_101_v07_auto.json", "mmlongbench_101_v071_v41flash.json", "blind_mmlongbench_91_v06.json"):
        p = res_dir / name
        if p.is_file():
            report["mmlongbench"][name] = analyze_mmlongbench(p)

    for name in ("contractnli_150_v06.json", "contractnli_150_v07_auto.json", "contractnli_150_v071_v41flash.json", "blind_contractnli_dev150_v06.json"):
        p = res_dir / name
        if p.is_file():
            report["contractnli"][name] = analyze_contractnli(p)

    out_path = res_dir / "paper_stats_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
