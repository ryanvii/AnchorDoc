"""Head-to-head MCP Agent Benchmark: Same LLM Agent (Native Full-PDF vs AnchorDoc Zero-LLM MCP).

Compares the EXACT same model (e.g. DeepSeek-V4.1-Flash, temperature=0) on real multi-hundred-page
SEC 10-K filings from FinanceBench under two arms:
  1) Native Full-PDF Stuffing: entire extracted PDF text stuffed into the 1M context window.
  2) Agent + AnchorDoc Zero-LLM 4-Tool MCP: standard OpenAI tool-calling loop over AnchorDocMCPApp
     (`doc_outline` -> `doc_locate` -> `doc_read` -> `doc_verify`) with zero internal LLM calls.

Both arms are graded by the exact same `judge_answer` evaluator from `run_financebench_full.py`
and checked deterministically for verbatim quote verification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchordoc.docbase import PdfDocument
from anchordoc.mcp_server import AnchorDocMCPApp, MCP_INSTRUCTIONS, TOOL_DEFINITIONS
from anchordoc.runtime import CachedLLM, verify_quote
from anchordoc.store.bundle import BundleStore
from benchmarks.run_financebench_full import judge_answer


def _resolve_api_key() -> str:
    key = os.environ.get("ANCHORDOC_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    key_file = os.environ.get("ANCHORDOC_LLM_KEY_FILE") or ""
    if not key and key_file:
        key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Set ANCHORDOC_LLM_KEY or ANCHORDOC_LLM_KEY_FILE.")
    return key


def _openai_tools_from_mcp() -> list[dict[str, Any]]:
    """Convert AnchorDoc MCP TOOL_DEFINITIONS to strict OpenAI function-calling schemas."""
    tools: list[dict[str, Any]] = []
    for td in TOOL_DEFINITIONS:
        schema = json.loads(json.dumps(td["inputSchema"]))
        # Normalize oneOf for broad OpenAI-compatible tool-calling endpoints
        props = schema.get("properties") or {}
        if "quotes" in props:
            props["quotes"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verbatim quote strings copied directly from doc_read.",
            }
        tools.append({
            "type": "function",
            "function": {
                "name": td["name"],
                "description": td["description"],
                "parameters": schema,
            },
        })
    return tools


def _extract_usage(resp: Any) -> tuple[int, int, int]:
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0, 0
    pt = int(getattr(u, "prompt_tokens", 0) or 0)
    ct = int(getattr(u, "completion_tokens", 0) or 0)
    cached = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
    if not cached:
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
    return pt, ct, cached


def _select_stratified_samples(
    samples_path: Path, pdf_dir: Path, per_type: int = 5
) -> list[tuple[dict[str, Any], Path, int]]:
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    buckets: dict[str, list[tuple[dict[str, Any], Path, int]]] = {
        "metrics-generated": [],
        "novel-generated": [],
        "domain-relevant": [],
    }
    page_counts: dict[str, int] = {}
    used_docs_per_type: dict[str, set[str]] = {k: set() for k in buckets}
    for row in rows:
        qt = str(row.get("question_type") or "")
        if qt not in buckets or len(buckets[qt]) >= per_type:
            continue
        doc_name = str(row.get("doc_name") or "")
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.is_file():
            continue
        if doc_name in used_docs_per_type[qt]:
            continue
        if doc_name not in page_counts:
            doc = PdfDocument.from_pdf(pdf_path)
            page_counts[doc_name] = doc.num_pages
        n_pages = page_counts[doc_name]
        if n_pages < 40:
            continue
        buckets[qt].append((row, pdf_path, n_pages))
        used_docs_per_type[qt].add(doc_name)
    out: list[tuple[dict[str, Any], Path, int]] = []
    for qt in ("metrics-generated", "novel-generated", "domain-relevant"):
        out.extend(buckets[qt])
    return out


def run_native_full_pdf(
    client: OpenAI,
    model: str,
    doc: PdfDocument,
    question: str,
    max_chars: int = 450_000,
) -> dict[str, Any]:
    """Arm 1: Stuff the entire extracted PDF text directly into the model's context window."""
    t0 = time.perf_counter()
    parts: list[str] = []
    total = 0
    truncated = False
    for p in range(1, doc.num_pages + 1):
        block = f"[Page {p}]\n{doc.page_text(p)}"
        if total + len(block) > max_chars:
            truncated = True
            break
        parts.append(block)
        total += len(block)
    full_text = "\n\n".join(parts)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=900,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior SEC financial analyst. Answer the user's question strictly from the "
                    "provided full filing text. Return JSON only: "
                    '{"can_answer": true/false, "answer": "concise final answer with units", '
                    '"verbatim_quote": "exact substring from filing", "cited_pages": [int]}'
                ),
            },
            {
                "role": "user",
                "content": f"Full SEC Filing ({doc.num_pages} pages):\n\n{full_text}\n\nQuestion: {question}",
            },
        ],
    )
    elapsed = time.perf_counter() - t0
    pt, ct, cached = _extract_usage(resp)
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"can_answer": bool(raw), "answer": raw, "verbatim_quote": "", "cited_pages": []}

    ans = str(parsed.get("answer") or "").strip()
    quote = str(parsed.get("verbatim_quote") or "").strip()
    cited = [int(p) for p in (parsed.get("cited_pages") or []) if isinstance(p, int) and 1 <= p <= doc.num_pages]
    # Deterministically check verbatim quote on cited pages (or full doc fallback)
    q_verified = False
    if quote:
        scope_pages = cited or list(range(1, doc.num_pages + 1))
        scope_text = "\n".join(doc.page_text(p) for p in scope_pages)
        q_verified = verify_quote(quote, scope_text)[0]

    return {
        "can_answer": bool(parsed.get("can_answer", True)),
        "answer": ans,
        "verbatim_quote": quote,
        "cited_pages": cited,
        "quote_verified": q_verified,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cached_tokens": cached,
        "llm_turns": 1,
        "tool_calls": 0,
        "truncated": truncated,
        "elapsed_sec": round(elapsed, 2),
    }


def run_mcp_agent(
    client: OpenAI,
    model: str,
    mcp_app: AnchorDocMCPApp,
    tools: list[dict[str, Any]],
    pdf_path: Path,
    doc: PdfDocument,
    question: str,
    max_turns: int = 10,
) -> dict[str, Any]:
    """Arm 2: Same model equipped with AnchorDoc's 4 Zero-LLM progressive MCP tools."""
    t0 = time.perf_counter()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{MCP_INSTRUCTIONS}\n\n"
                "You are a senior SEC financial analyst using AnchorDoc MCP tools to inspect a PDF filing.\n"
                "Workflow:\n"
                "1. Call doc_outline(path=...) first to see structure and doc_id.\n"
                "2. Call doc_locate(doc_id=..., query=...) with targeted financial terms to find candidate pages.\n"
                "3. Call doc_read(doc_id=..., pages=[...]) on the top relevant pages (<=5 pages).\n"
                "4. Before answering, ALWAYS call doc_verify(doc_id=..., quotes=[...], formula=..., cited_pages=[...]) "
                "to deterministically verify your verbatim quote(s) and arithmetic.\n"
                "5. When ready, respond with a final JSON object (no markdown fences): "
                '{"can_answer": true/false, "answer": "concise final answer with exact units", '
                '"verbatim_quote": "verified quote", "cited_pages": [int]}'
            ),
        },
        {
            "role": "user",
            "content": f"PDF path: {pdf_path}\nQuestion: {question}",
        },
    ]

    total_pt = 0
    total_ct = 0
    total_cached = 0
    llm_turns = 0
    tool_counts: Counter[str] = Counter()
    verify_quote_ok = False
    verify_calc_used = False
    final_raw = ""

    for turn in range(max_turns):
        llm_turns += 1
        use_tools = tools if turn < max_turns - 1 else None
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "max_tokens": 900,
            "messages": messages,
        }
        if use_tools is not None:
            kwargs["tools"] = use_tools
            kwargs["tool_choice"] = "auto"
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Tool budget reached. Respond now with the final JSON object ONLY: "
                    '{"can_answer": true/false, "answer": "...", "verbatim_quote": "...", "cited_pages": [int]}'
                ),
            })

        resp = client.chat.completions.create(**kwargs)
        pt, ct, cached = _extract_usage(resp)
        total_pt += pt
        total_ct += ct
        total_cached += cached

        msg = resp.choices[0].message
        tcalls = getattr(msg, "tool_calls", None) or []
        if tcalls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tcalls
                ],
            })
            for tc in tcalls:
                fname = tc.function.name
                tool_counts[fname] += 1
                try:
                    fargs = json.loads(tc.function.arguments or "{}")
                except Exception:
                    fargs = {}
                try:
                    res = mcp_app.call_tool(fname, fargs)
                except Exception as exc:
                    res = {"error": str(exc)}
                if fname == "doc_verify" and isinstance(res, dict):
                    q_res = res.get("quotes") or []
                    if any(isinstance(q, dict) and q.get("verified") for q in q_res):
                        verify_quote_ok = True
                    if res.get("calc_value") is not None:
                        verify_calc_used = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(res, ensure_ascii=False),
                })
            continue

        final_raw = (msg.content or "").strip()
        break

    elapsed = time.perf_counter() - t0
    cleaned = final_raw
    if "```" in cleaned:
        cleaned = cleaned.split("```")[-2] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        parsed = json.loads(cleaned.strip())
    except Exception:
        # Try extracting first JSON object
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            try:
                parsed = json.loads(cleaned[s : e + 1])
            except Exception:
                parsed = {"can_answer": bool(final_raw), "answer": final_raw, "verbatim_quote": "", "cited_pages": []}
        else:
            parsed = {"can_answer": bool(final_raw), "answer": final_raw, "verbatim_quote": "", "cited_pages": []}

    ans = str(parsed.get("answer") or "").strip()
    quote = str(parsed.get("verbatim_quote") or "").strip()
    cited = [int(p) for p in (parsed.get("cited_pages") or []) if isinstance(p, int) and 1 <= p <= doc.num_pages]

    q_verified = verify_quote_ok
    if not q_verified and quote:
        scope_pages = cited or list(range(1, doc.num_pages + 1))
        scope_text = "\n".join(doc.page_text(p) for p in scope_pages)
        q_verified = verify_quote(quote, scope_text)[0]

    return {
        "can_answer": bool(parsed.get("can_answer", True)),
        "answer": ans,
        "verbatim_quote": quote,
        "cited_pages": cited,
        "quote_verified": q_verified,
        "verify_tool_quote_ok": verify_quote_ok,
        "verify_tool_calc_used": verify_calc_used,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "cached_tokens": total_cached,
        "llm_turns": llm_turns,
        "tool_calls": sum(tool_counts.values()),
        "tool_breakdown": dict(tool_counts),
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Head-to-head MCP Agent vs Native Full-PDF benchmark.")
    ap.add_argument("--per-type", type=int, default=5, help="Questions per FinanceBench category (default 5 -> 15 total)")
    ap.add_argument("--out", default="results/mcp_agent_benchmark_v08.json")
    ap.add_argument("--judge-cache", default="cache/mcp_agent_judge_v08")
    args = ap.parse_args()

    base_url = os.environ.get("ANCHORDOC_LLM_BASE", "https://api.deepseek.com/v1")
    model = os.environ.get("ANCHORDOC_LLM_MODEL", "deepseek-flash")
    api_key = _resolve_api_key()

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=180.0,
        http_client=httpx.Client(trust_env=False, timeout=180.0),
    )
    judge_llm = CachedLLM(cache_dir=Path(args.judge_cache), model=model, base_url=base_url, api_key=api_key)

    pdf_dir = ROOT / "data" / "financebench_pdfs"
    samples_path = ROOT / "data" / "financebench_merged.jsonl"
    selected = _select_stratified_samples(samples_path, pdf_dir, per_type=args.per_type)
    print(f"MCP Agent Benchmark: {len(selected)} stratified SEC 10-K questions | model={model}")

    tools = _openai_tools_from_mcp()
    results_per_item: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="anchordoc_mcp_bench_") as tmp_bundle:
        mcp_app = AnchorDocMCPApp(roots=[pdf_dir], store=BundleStore(Path(tmp_bundle)))
        for idx, (row, pdf_path, n_pages) in enumerate(selected, start=1):
            doc = PdfDocument.from_pdf(pdf_path)
            q = row["question"]
            gold = row["answer"]
            qtype = row["question_type"]
            doc_name = row["doc_name"]

            native = run_native_full_pdf(client, model, doc, q)
            j_native = judge_answer(q, gold, native["answer"], native["can_answer"], judge_llm)

            mcp = run_mcp_agent(client, model, mcp_app, tools, pdf_path, doc, q)
            j_mcp = judge_answer(q, gold, mcp["answer"], mcp["can_answer"], judge_llm)

            tok_save = round((1.0 - mcp["prompt_tokens"] / max(1, native["prompt_tokens"])) * 100.0, 1)
            print(
                f"[{idx:02d}/{len(selected)}] {doc_name} ({n_pages}p, {qtype[:6]}) | "
                f"Native: {j_native['verdict']:9s} {native['prompt_tokens']:7d} tok (q_ok={native['quote_verified']}) | "
                f"MCP: {j_mcp['verdict']:9s} {mcp['prompt_tokens']:6d} tok ({tok_save:+5.1f}%) "
                f"tools={mcp['tool_calls']} (q_ok={mcp['quote_verified']})"
            )
            results_per_item.append({
                "i": idx,
                "financebench_id": row.get("financebench_id"),
                "doc_name": doc_name,
                "num_pages": n_pages,
                "question_type": qtype,
                "question": q,
                "gold": gold,
                "native_full_pdf": {**native, "judge": j_native},
                "anchordoc_mcp_agent": {**mcp, "judge": j_mcp},
            })

    def _agg(arm: str) -> dict[str, Any]:
        n = len(results_per_item)
        corr = sum(1 for r in results_per_item if r[arm]["judge"]["verdict"] == "correct")
        inc = sum(1 for r in results_per_item if r[arm]["judge"]["verdict"] == "incorrect")
        abst = sum(1 for r in results_per_item if r[arm]["judge"]["verdict"] == "abstain")
        q_ok = sum(1 for r in results_per_item if r[arm]["quote_verified"])
        avg_pt = sum(r[arm]["prompt_tokens"] for r in results_per_item) / n
        avg_ct = sum(r[arm]["completion_tokens"] for r in results_per_item) / n
        avg_cached = sum(r[arm]["cached_tokens"] for r in results_per_item) / n
        avg_turns = sum(r[arm]["llm_turns"] for r in results_per_item) / n
        avg_tools = sum(r[arm]["tool_calls"] for r in results_per_item) / n
        avg_lat = sum(r[arm]["elapsed_sec"] for r in results_per_item) / n
        by_type = {}
        for qt in ("metrics-generated", "novel-generated", "domain-relevant"):
            sub = [r for r in results_per_item if r["question_type"] == qt]
            if sub:
                by_type[qt] = round(sum(1 for r in sub if r[arm]["judge"]["verdict"] == "correct") / len(sub) * 100, 1)
        return {
            "n": n,
            "e2e_acc_%": round(corr / n * 100, 1),
            "incorrect_%": round(inc / n * 100, 1),
            "abstain_%": round(abst / n * 100, 1),
            "quote_verified_%": round(q_ok / n * 100, 1),
            "avg_prompt_tokens": round(avg_pt, 0),
            "avg_completion_tokens": round(avg_ct, 0),
            "avg_cached_tokens": round(avg_cached, 0),
            "avg_llm_turns": round(avg_turns, 2),
            "avg_tool_calls": round(avg_tools, 2),
            "avg_elapsed_sec": round(avg_lat, 2),
            "by_type_acc_%": by_type,
        }

    s_native = _agg("native_full_pdf")
    s_mcp = _agg("anchordoc_mcp_agent")
    token_reduction = round((1.0 - s_mcp["avg_prompt_tokens"] / max(1.0, s_native["avg_prompt_tokens"])) * 100.0, 1)

    summary = {
        "benchmark": f"FinanceBench MCP Agent Head-to-Head ({len(selected)} stratified SEC 10-K questions)",
        "model": model,
        "base_url": base_url,
        "avg_doc_pages": round(sum(r["num_pages"] for r in results_per_item) / len(results_per_item), 1),
        "native_full_pdf": s_native,
        "anchordoc_mcp_agent": s_mcp,
        "prompt_token_reduction_%": token_reduction,
        "per_item": results_per_item,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== MCP Agent Head-to-Head Summary ===")
    print(f"  Avg PDF Pages      : {summary['avg_doc_pages']} pages")
    print(f"  Native Full-PDF    : Acc={s_native['e2e_acc_%']}% | QuoteVer={s_native['quote_verified_%']}% | AvgPromptTok={int(s_native['avg_prompt_tokens'])} | Lat={s_native['avg_elapsed_sec']}s")
    print(f"  AnchorDoc MCP Agent: Acc={s_mcp['e2e_acc_%']}% | QuoteVer={s_mcp['quote_verified_%']}% | AvgPromptTok={int(s_mcp['avg_prompt_tokens'])} ({token_reduction:+.1f}% saved) | Tools={s_mcp['avg_tool_calls']} | Lat={s_mcp['avg_elapsed_sec']}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
