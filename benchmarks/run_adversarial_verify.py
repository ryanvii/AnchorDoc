"""Zero-API-key, clean-clone-ready Adversarial Verification Mini-Benchmark (24 cases).

Directly tests AnchorDoc v0.9.0's independent claim verification layer against a naive
alphanumeric-normalization verifier on traps that silently fool ordinary citation checks:
  1. Sign flips (`-12%` -> `12%`, even when `+12%` appears elsewhere on the same page)
  2. Accounting parenthetical negatives (`$(145)` -> `$145`, `(8.4%)` -> `8.4%`)
  3. Fiscal period / year column swaps (all 4 numbers exist on the page, wrong year grabbed)
  4. Row / line-item swaps (grabbing Net Income when claiming Operating Income)
  5. Million vs Billion vs Thousand unit scale confusion
  6. Percentage points (`pp`) vs relative percent (`%`) confusion
  7. Arithmetic formula recomputation errors & sign flips
  8. 10 valid ground-truth claims (including unit conversion `$1,200M == $1.2B` and accounting
     negative equivalence `$(145)` == `-145`) to measure False-Alarm Rate!

Run from any clean clone in ~1 second (0 downloads, 0 LLM API keys):
    python3 benchmarks/run_adversarial_verify.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchordoc import PdfDocument, safe_calc, verify_claim  # noqa: E402


# ---------------------------------------------------------------------------
# Self-contained minimal valid multi-page PDF writer (zero extra dependencies)
# ---------------------------------------------------------------------------

def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_sample_filing_pdf(dest: Path) -> Path:
    """Create a realistic 3-page corporate annual financial filing PDF for reproducible offline verification."""
    pages_lines: list[list[str]] = [
        [
            "ACME INDUSTRIAL CORP. - FY2024 ANNUAL FINANCIAL REPORT",
            "Item 7. Management Discussion and Analysis",
            "Executive Summary:",
            "Total revenue growth was 12% year-over-year driven by North America demand.",
            "However, EMEA segment Operating margin was -12% due to restructuring charges.",
            "Consolidated net loss in Q4 was $(145) million compared to $210 million profit in Q3.",
            "Adjusted EBITDA margin declined to (8.4%) in the divested legacy division.",
            "All tabular financial figures on Page 2 are reported in millions unless stated otherwise.",
        ],
        [
            "CONSOLIDATED STATEMENTS OF OPERATIONS (in millions, except percentages)",
            "Line Item                             FY2024      FY2023      FY2022",
            "Total Revenue                          1,200       1,050         980",
            "Cost of Goods Sold                       720         672         637",
            "Gross Profit                             480         378         343",
            "Operating Income                         180         147         120",
            "Restructuring Expense                  (145)        (40)        (15)",
            "Net Income                               125         110          95",
            "Gross Margin                           40.0%       36.0%       35.0%",
            "Operating Margin                       15.0%       14.0%       12.2%",
            "EMEA Segment Margin                   (12.0%)       3.5%        5.1%",
        ],
        [
            "BALANCE SHEET & DEBT COVENANT SCHEDULE (in billions)",
            "Metric                                FY2024      FY2023",
            "Total Assets                            4.80        4.40",
            "Total Liabilities                       2.64        2.20",
            "Long-Term Debt                          1.85        1.60",
            "Cash and Equivalents                    0.95        0.80",
            "Debt-to-Asset Ratio                    55.0%       50.0%",
            "Covenant Limit: Debt-to-Asset Ratio shall not exceed 65.0% at fiscal year-end.",
        ],
    ]

    objects: list[bytes] = []

    def add_obj(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    # Reserve 1: Catalog, 2: Pages, 3: Font
    add_obj(b"")
    add_obj(b"")
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_ids: list[int] = []
    for lines in pages_lines:
        stream_cmds = ["BT", "/F1 10 Tf", "40 760 Td", "16 TL"]
        for i, line in enumerate(lines):
            if i == 0:
                stream_cmds.append(f"({_pdf_escape(line)}) Tj")
            else:
                stream_cmds.append(f"T* ({_pdf_escape(line)}) Tj")
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin-1")
        content_id = add_obj(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream"
        )
        page_id = add_obj(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode("latin-1")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("latin-1"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    dest.write_bytes(bytes(out))
    return dest


# ---------------------------------------------------------------------------
# Naive baseline verifier (representing v0.8 / typical alphanumeric citation checks)
# ---------------------------------------------------------------------------

def _naive_norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def naive_verify(doc: PdfDocument, case: dict[str, Any]) -> bool:
    """Typical citation & loose number-existence check (strips punctuation/signs, ignores row/period/unit)."""
    full_norm = _naive_norm("\n".join(doc.pages))
    full_nums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", "\n".join(doc.pages))}

    for q in case.get("quotes") or ():
        qt = q["text"] if isinstance(q, dict) else str(q)
        if _naive_norm(qt) not in full_norm:
            return False

    for op in case.get("operands") or ():
        v = abs(float(op["value"]))
        if not any(
            abs(s * f - v) <= 0.5 or (v > 0 and abs(s * f - v) / v < 0.002)
            for s in full_nums
            for f in (1.0, 1e3, 1e-3, 1e6, 1e-6, 100.0, 0.01)
        ):
            return False

    formula = case.get("formula") or ""
    if formula:
        val = safe_calc(formula)
        cv = case.get("claimed_value")
        if val is None:
            return False
        if cv is not None and abs(val - float(cv)) > 0.05:
            return False
    return True


# ---------------------------------------------------------------------------
# 24 Adversarial & Ground-Truth Verification Cases (14 Traps + 10 Valid)
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    # --- 14 ADVERSARIAL TRAPS (expected_verified = False) ---
    {
        "id": "ADV-01",
        "category": "quote_sign_flip",
        "description": "Sign flip in quote (`-12%` rewritten as `12%`) while `+12%` ALSO exists on page 1!",
        "expected_verified": False,
        "expected_error": "QUOTE_SIGN_FLIP",
        "quotes": [{"text": "EMEA segment Operating margin was 12% due to restructuring charges.", "page": 1}],
    },
    {
        "id": "ADV-02",
        "category": "quote_sign_flip",
        "description": "Accounting parenthetical loss `$(145)` silently flipped to positive `$145` in quote.",
        "expected_verified": False,
        "expected_error": "QUOTE_SIGN_FLIP",
        "quotes": [{"text": "Consolidated net loss in Q4 was $145 million compared to $210 million", "page": 1}],
    },
    {
        "id": "ADV-03",
        "category": "quote_sign_flip",
        "description": "Parenthetical negative margin `(8.4%)` flipped to positive `8.4%` in quote.",
        "expected_verified": False,
        "expected_error": "QUOTE_SIGN_FLIP",
        "quotes": [{"text": "Adjusted EBITDA margin declined to 8.4% in the divested legacy division.", "page": 1}],
    },
    {
        "id": "ADV-04",
        "category": "operand_sign_flip",
        "description": "Restructuring Expense is `(145)` (-145M) in FY2024, AI claims positive `+145`.",
        "expected_verified": False,
        "expected_error": "SIGN_FLIP",
        "operands": [{"name": "FY2024 Restructuring", "value": 145, "unit": "million", "row_label": "Restructuring Expense", "period": "FY2024", "page": 2}],
    },
    {
        "id": "ADV-05",
        "category": "operand_sign_flip",
        "description": "EMEA Segment Margin is `(12.0%)` in FY2024, AI claims `+12.0%`.",
        "expected_verified": False,
        "expected_error": "SIGN_FLIP",
        "operands": [{"name": "FY2024 EMEA Margin", "value": 12.0, "unit": "%", "row_label": "EMEA Segment Margin", "period": "FY2024", "page": 2}],
    },
    {
        "id": "ADV-06",
        "category": "period_mismatch",
        "description": "Year-column swap: AI uses FY2023 Gross Profit (378) while claiming FY2024 (480) — both exist on Page 2!",
        "expected_verified": False,
        "expected_error": "PERIOD_MISMATCH",
        "operands": [
            {"name": "FY2024 Revenue", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "FY2024 Gross Profit", "value": 378, "unit": "million", "row_label": "Gross Profit", "period": "FY2024", "page": 2},
        ],
        "formula": "378 / 1200 * 100",
        "claimed_value": 31.5,
        "claimed_unit": "%",
    },
    {
        "id": "ADV-07",
        "category": "period_mismatch",
        "description": "Year-column swap on Balance Sheet: AI cites FY2023 Total Assets (4.40B) as FY2024.",
        "expected_verified": False,
        "expected_error": "PERIOD_MISMATCH",
        "operands": [{"name": "FY2024 Assets", "value": 4.40, "unit": "billion", "row_label": "Total Assets", "period": "FY2024", "page": 3}],
    },
    {
        "id": "ADV-08",
        "category": "period_mismatch",
        "description": "AI grabs FY2022 Operating Margin (12.2%) claiming it is FY2023 (14.0%).",
        "expected_verified": False,
        "expected_error": "PERIOD_MISMATCH",
        "operands": [{"name": "FY2023 Op Margin", "value": 12.2, "unit": "%", "row_label": "Operating Margin", "period": "FY2023", "page": 2}],
    },
    {
        "id": "ADV-09",
        "category": "unit_scale_mismatch",
        "description": "Million vs Billion confusion: Page 2 is in millions (1,200M), AI claims Total Revenue is $1,200 billion.",
        "expected_verified": False,
        "expected_error": "UNIT_SCALE_MISMATCH",
        "operands": [{"name": "FY2024 Revenue", "value": 1200, "unit": "billion", "row_label": "Total Revenue", "period": "FY2024", "page": 2}],
    },
    {
        "id": "ADV-10",
        "category": "unit_scale_mismatch",
        "description": "Billion vs Million confusion: Page 3 is in billions (4.80B), AI claims Total Assets is $4.80 million.",
        "expected_verified": False,
        "expected_error": "UNIT_SCALE_MISMATCH",
        "operands": [{"name": "FY2024 Assets", "value": 4.80, "unit": "million", "row_label": "Total Assets", "period": "FY2024", "page": 3}],
    },
    {
        "id": "ADV-11",
        "category": "pp_vs_pct_confusion",
        "description": "Gross Margin rose from 36.0% to 40.0%: AI computes `40.0 - 36.0 = 4.0` but labels it `%` instead of `pp`.",
        "expected_verified": False,
        "expected_error": "PP_VS_PCT_CONFUSION",
        "claim_text": "Gross margin increased by 4.0% from FY2023 to FY2024.",
        "operands": [
            {"name": "FY2024 Gross Margin", "value": 40.0, "unit": "%", "row_label": "Gross Margin", "period": "FY2024", "page": 2},
            {"name": "FY2023 Gross Margin", "value": 36.0, "unit": "%", "row_label": "Gross Margin", "period": "FY2023", "page": 2},
        ],
        "formula": "40.0 - 36.0",
        "claimed_value": 4.0,
        "claimed_unit": "%",
    },
    {
        "id": "ADV-12",
        "category": "pp_vs_pct_confusion",
        "description": "Relative growth `(1200 - 1050) / 1050 * 100 = 14.29%` mislabeled as `14.29 pp`.",
        "expected_verified": False,
        "expected_error": "PP_VS_PCT_CONFUSION",
        "claim_text": "Revenue grew by 14.29 pp.",
        "operands": [
            {"name": "FY2024 Revenue", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "FY2023 Revenue", "value": 1050, "unit": "million", "row_label": "Total Revenue", "period": "FY2023", "page": 2},
        ],
        "formula": "(1200 - 1050) / 1050 * 100",
        "claimed_value": 14.29,
        "claimed_unit": "pp",
    },
    {
        "id": "ADV-13",
        "category": "row_label_mismatch",
        "description": "Row swap: AI grabs Net Income (125) on Page 2 but labels it as FY2024 Operating Income (180).",
        "expected_verified": False,
        "expected_error": "ROW_LABEL_MISMATCH",
        "operands": [{"name": "FY2024 Operating Income", "value": 125, "unit": "million", "row_label": "Operating Income", "period": "FY2024", "page": 2}],
    },
    {
        "id": "ADV-14",
        "category": "formula_mismatch",
        "description": "Mental math error: operands grounded (`1200 - 720`), but AI claims Gross Profit is `520` instead of `480`.",
        "expected_verified": False,
        "expected_error": "FORMULA_MISMATCH",
        "operands": [
            {"name": "FY2024 Revenue", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "FY2024 COGS", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        "formula": "1200 - 720",
        "claimed_value": 520,
        "claimed_unit": "million",
    },
    # --- 10 VALID GROUND-TRUTH CLAIMS (expected_verified = True; measures False-Alarm Rate) ---
    {
        "id": "VAL-01",
        "category": "valid_negative_quote",
        "description": "Exact negative quote `-12%` verified cleanly.",
        "expected_verified": True,
        "quotes": [{"text": "EMEA segment Operating margin was -12% due to restructuring charges.", "page": 1}],
    },
    {
        "id": "VAL-02",
        "category": "valid_accounting_negative",
        "description": "Accounting parenthetical loss `$(145)` accurately cited as `-$145`.",
        "expected_verified": True,
        "quotes": [{"text": "Consolidated net loss in Q4 was -$145 million compared to $210 million", "page": 1}],
    },
    {
        "id": "VAL-03",
        "category": "valid_gross_profit",
        "description": "FY2024 Gross Profit recomputation `1200 - 720 = 480` (in millions) with exact row & period binding.",
        "expected_verified": True,
        "operands": [
            {"name": "FY2024 Revenue", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "FY2024 COGS", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        "formula": "1200 - 720",
        "claimed_value": 480,
        "claimed_unit": "million",
    },
    {
        "id": "VAL-04",
        "category": "valid_pp_expansion",
        "description": "Gross Margin expansion `40.0 - 36.0 = 4.0 pp` properly labeled in percentage points (`pp`).",
        "expected_verified": True,
        "claim_text": "Gross margin expanded by 4.0 percentage points from FY2023 to FY2024.",
        "operands": [
            {"name": "FY2024 Gross Margin", "value": 40.0, "unit": "%", "row_label": "Gross Margin", "period": "FY2024", "page": 2},
            {"name": "FY2023 Gross Margin", "value": 36.0, "unit": "%", "row_label": "Gross Margin", "period": "FY2023", "page": 2},
        ],
        "formula": "40.0 - 36.0",
        "claimed_value": 4.0,
        "claimed_unit": "pp",
    },
    {
        "id": "VAL-05",
        "category": "valid_relative_growth",
        "description": "FY2024 Revenue YoY growth `(1200 - 1050) / 1050 * 100 = 14.29%` properly labeled `%`.",
        "expected_verified": True,
        "operands": [
            {"name": "FY2024 Revenue", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "FY2023 Revenue", "value": 1050, "unit": "million", "row_label": "Total Revenue", "period": "FY2023", "page": 2},
        ],
        "formula": "(1200 - 1050) / 1050 * 100",
        "claimed_value": 14.29,
        "claimed_unit": "%",
    },
    {
        "id": "VAL-06",
        "category": "valid_negative_operand",
        "description": "Accounting parenthetical `(145)` on Page 2 accurately bound as `-145` million.",
        "expected_verified": True,
        "operands": [{"name": "FY2024 Restructuring", "value": -145, "unit": "million", "row_label": "Restructuring Expense", "period": "FY2024", "page": 2}],
    },
    {
        "id": "VAL-07",
        "category": "valid_negative_margin",
        "description": "Accounting parenthetical `(12.0%)` on Page 2 accurately bound as `-12.0%`.",
        "expected_verified": True,
        "operands": [{"name": "FY2024 EMEA Margin", "value": -12.0, "unit": "%", "row_label": "EMEA Segment Margin", "period": "FY2024", "page": 2}],
    },
    {
        "id": "VAL-08",
        "category": "valid_unit_conversion",
        "description": "Legitimate unit conversion: table says `1,200` (in millions), claim states `1.2` billion.",
        "expected_verified": True,
        "operands": [{"name": "FY2024 Revenue", "value": 1.2, "unit": "billion", "row_label": "Total Revenue", "period": "FY2024", "page": 2}],
    },
    {
        "id": "VAL-09",
        "category": "valid_debt_ratio",
        "description": "FY2024 Debt-to-Asset ratio `2.64 / 4.80 * 100 = 55.0%` (in billions) verified on Page 3.",
        "expected_verified": True,
        "operands": [
            {"name": "FY2024 Liabilities", "value": 2.64, "unit": "billion", "row_label": "Total Liabilities", "period": "FY2024", "page": 3},
            {"name": "FY2024 Assets", "value": 4.80, "unit": "billion", "row_label": "Total Assets", "period": "FY2024", "page": 3},
        ],
        "formula": "2.64 / 4.80 * 100",
        "claimed_value": 55.0,
        "claimed_unit": "%",
    },
    {
        "id": "VAL-10",
        "category": "valid_covenant_quote",
        "description": "Exact debt covenant quote on Page 3 verified cleanly.",
        "expected_verified": True,
        "quotes": [{"text": "Debt-to-Asset Ratio shall not exceed 65.0% at fiscal year-end.", "page": 3}],
    },
]


def run_benchmark(out_json: Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = build_sample_filing_pdf(Path(td) / "acme_fy2024_sample.pdf")
        doc = PdfDocument.from_pdf(pdf_path)

        traps = [c for c in CASES if not c["expected_verified"]]
        valids = [c for c in CASES if c["expected_verified"]]

        naive_false_pass = 0
        naive_false_alarm = 0
        ad_false_pass = 0
        ad_false_alarm = 0
        ad_exact_error = 0
        details: list[dict[str, Any]] = []

        for c in CASES:
            n_ok = naive_verify(doc, c)
            ad_res = verify_claim(
                doc,
                claim_text=c.get("claim_text", ""),
                quotes=c.get("quotes") or (),
                formula=c.get("formula", ""),
                claimed_value=c.get("claimed_value"),
                claimed_unit=c.get("claimed_unit", ""),
                operands=c.get("operands") or (),
                cited_pages=c.get("cited_pages") or (),
                strict_scale=True,
            )
            a_ok = ad_res["verified"]
            exp = c["expected_verified"]
            if not exp:
                if n_ok:
                    naive_false_pass += 1
                if a_ok:
                    ad_false_pass += 1
                elif c.get("expected_error") in ad_res["error_codes"]:
                    ad_exact_error += 1
            else:
                if not n_ok:
                    naive_false_alarm += 1
                if not a_ok:
                    ad_false_alarm += 1

            details.append({
                "id": c["id"],
                "category": c["category"],
                "description": c["description"],
                "expected_verified": exp,
                "naive_verified": n_ok,
                "anchordoc_verified": a_ok,
                "anchordoc_errors": ad_res["error_codes"],
                "diagnostics": ad_res["diagnostics"],
                "citations": ad_res["citations"],
            })

    summary = {
        "benchmark": "AnchorDoc v0.9.0 Adversarial Financial Verification Mini-Bench (0 API keys)",
        "total_cases": len(CASES),
        "adversarial_traps": len(traps),
        "valid_claims": len(valids),
        "naive_verifier": {
            "trap_catch_rate_%": round((len(traps) - naive_false_pass) / len(traps) * 100, 1),
            "false_pass_rate_%": round(naive_false_pass / len(traps) * 100, 1),
            "false_pass_count": naive_false_pass,
            "false_alarm_rate_%": round(naive_false_alarm / len(valids) * 100, 1),
            "false_alarm_count": naive_false_alarm,
        },
        "anchordoc_v09": {
            "trap_catch_rate_%": round((len(traps) - ad_false_pass) / len(traps) * 100, 1),
            "false_pass_rate_%": round(ad_false_pass / len(traps) * 100, 1),
            "false_pass_count": ad_false_pass,
            "exact_error_code_match": f"{ad_exact_error}/{len(traps)}",
            "false_alarm_rate_%": round(ad_false_alarm / len(valids) * 100, 1),
            "false_alarm_count": ad_false_alarm,
        },
        "details": details,
    }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Run AnchorDoc v0.9.0 Adversarial Verification Mini-Bench (0 API keys).")
    ap.add_argument("--out", default="", help="Optional path to save JSON results.")
    args = ap.parse_args()
    res = run_benchmark(Path(args.out) if args.out else None)
    nv = res["naive_verifier"]
    ad = res["anchordoc_v09"]
    print(f"\n=== {res['benchmark']} ===")
    print(f"Cases: {res['total_cases']} ({res['adversarial_traps']} adversarial traps + {res['valid_claims']} valid claims)")
    print(f"Naive Citation/Number Check -> Catch Rate: {nv['trap_catch_rate_%']:5.1f}% | False-Pass: {nv['false_pass_rate_%']:5.1f}% ({nv['false_pass_count']}/{res['adversarial_traps']}) | False-Alarm: {nv['false_alarm_rate_%']:5.1f}%")
    print(f"AnchorDoc v0.9.0 Verifier   -> Catch Rate: {ad['trap_catch_rate_%']:5.1f}% | False-Pass: {ad['false_pass_rate_%']:5.1f}% ({ad['false_pass_count']}/{res['adversarial_traps']}) | False-Alarm: {ad['false_alarm_rate_%']:5.1f}% | Exact Diag: {ad['exact_error_code_match']}")
    print("\nPer-case breakdown:")
    for d in res["details"]:
        tag = "TRAP " if not d["expected_verified"] else "VALID"
        n_mark = "PASS" if d["naive_verified"] else "FAIL"
        a_mark = "PASS" if d["anchordoc_verified"] else f"CATCH {d['anchordoc_errors']}"
        print(f"  [{d['id']}] ({tag}) Naive={n_mark:4s} | AnchorDoc={a_mark:34s} | {d['description']}")
    return 0 if (ad["false_pass_count"] == 0 and ad["false_alarm_count"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
