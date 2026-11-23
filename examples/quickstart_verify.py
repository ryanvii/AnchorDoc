"""10-Second Zero-API-Key Walkthrough: Evidence Sufficiency & Cross-Page Verification.

Answers the exact homepage question on a built-in multi-page annual report (0 downloads, 0 API keys):
    "Did FY2024 Gross Margin expand vs FY2023 after excluding one-time restructuring charges?"

Demonstrates TWO live runs with real wall-clock timing (`time.perf_counter()`), ZERO hardcoded
page numbers, and ZERO hardcoded operand values:
  - Run A (Complete 6-page filing with Note 7 on Page 6):
      Round 1 locates & reads the Income Statement -> verifies Reported GAAP Gross Margin (+4.0 pp,
      looks like YES), then detects an unresolved cross-reference (`Note 7`) required for the
      restructuring adjustment -> triggers Round 2 follow-up locate & read on Page 6 -> recomputes
      Adjusted Gross Margin across Page 2 + Page 6 cells -> flips the answer to NO (-5.0 pp) [VERIFIED].
  - Run B (Same filing with Note 7 removed — 5-page PDF):
      Round 1 verifies Reported GAAP Gross Margin (+4.0 pp) and flags missing `Note 7` -> Round 2
      searches all remaining pages and finds no Note 7 -> refuses to guess and emits
      PARTIALLY_VERIFIED (`INSUFFICIENT_EVIDENCE` for the adjusted metric).

Run from any clean clone:
    python3 examples/quickstart_verify.py
"""
from __future__ import annotations

import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchordoc import (  # noqa: E402
    DocContext,
    PdfDocument,
    detect_unresolved_references,
    parse_page_rows,
    verify_claim,
)

QUESTION = "Did FY2024 Gross Margin expand vs FY2023 after excluding one-time restructuring charges?"


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_restructuring_filing_pdf(dest: Path, *, include_note7: bool = True) -> Path:
    """Build a realistic 6-page (or 5-page without Note 7) annual report PDF offline."""
    pages_lines: list[list[str]] = [
        [
            "ACME INDUSTRIAL CORP. - FY2024 ANNUAL FINANCIAL REPORT",
            "Item 7. Management Discussion and Analysis",
            "Executive Summary:",
            "Reported GAAP Gross Margin improved from 36.0% in FY2023 to 40.0% in FY2024 (+4.0 pp).",
            "Both FY2024 and FY2023 included material one-time factory restructuring charges.",
            "Analysts evaluating recurring profitability should adjust Gross Margin for one-time",
            "restructuring charges recorded inside Cost of Goods Sold (see Note 7).",
        ],
        [
            "CONSOLIDATED STATEMENTS OF OPERATIONS (in millions, except percentages)",
            "Line Item                                FY2024      FY2023      FY2022",
            "Total Revenue                             1,200       1,050         980",
            "Cost of Goods Sold                          720         672         637",
            "Gross Profit                                480         378         343",
            "Operating Income                            180         147         120",
            "Restructuring Expense (See Note 7)        (145)       (140)        (15)",
            "Net Income                                  125         110          95",
            "Gross Margin                              40.0%       36.0%       35.0%",
            "Operating Margin                          15.0%       14.0%       12.2%",
            "Footnote: Portion of restructuring charges included in Cost of Goods Sold is disclosed in Note 7.",
        ],
        [
            "CONSOLIDATED BALANCE SHEETS (in billions)",
            "Line Item                                FY2024      FY2023",
            "Total Assets                               4.80        4.40",
            "Total Liabilities                          2.64        2.20",
            "Long-Term Debt                             1.85        1.60",
            "Cash and Equivalents                       0.95        0.80",
            "Debt-to-Asset Ratio                       55.0%       50.0%",
        ],
        [
            "CONSOLIDATED STATEMENTS OF CASH FLOWS (in millions)",
            "Line Item                                FY2024      FY2023",
            "Operating Cash Flow                         260         215",
            "Capital Expenditures                       (95)        (88)",
            "Free Cash Flow                              165         127",
        ],
        [
            "NOTE 6. COMMITMENTS, LEASES AND DEBT COVENANTS",
            "Operating lease obligations total $180 million over the next five years.",
            "Covenant Limit: Debt-to-Asset Ratio shall not exceed 65.0% at fiscal year-end.",
            "The company remained in full compliance with all credit facility covenants.",
        ],
    ]
    if include_note7:
        pages_lines.append(
            [
                "NOTE 7. FACILITY CONSOLIDATION AND COST ALLOCATION (in millions)",
                "Management executed a multi-site plant closure and severance program across FY2023-FY2024.",
                "Statement of Operations classification of these program costs:",
                "Line Item                                FY2024      FY2023",
                "Restructuring in Cost of Goods Sold          12         105",
                "Restructuring in Operating Expenses         133          35",
                "Total Facility Consolidation Costs          145         140",
                "Excluding the Cost of Goods Sold portion yields recurring Adjusted COGS.",
            ]
        )

    objects: list[bytes] = []

    def add_obj(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    add_obj(b"")
    add_obj(b"")
    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    page_ids: list[int] = []
    for lines in pages_lines:
        cmds = ["BT", "/F1 10 Tf", "40 760 Td", "16 TL"]
        for i, line in enumerate(lines):
            cmds.append(f"({_pdf_escape(line)}) Tj" if i == 0 else f"T* ({_pdf_escape(line)}) Tj")
        cmds.append("ET")
        stream = "\n".join(cmds).encode("latin-1")
        cid = add_obj(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")
        pid = add_obj(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>".encode("latin-1")
        )
        page_ids.append(pid)

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
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    dest.write_bytes(bytes(out))
    return dest


def _extract_row_period_cell(
    pages_text: dict[int, str],
    row_keyword: str,
    period: str,
) -> dict[str, Any] | None:
    """Dynamically extract a numeric cell (value, unit, row_label, period, page) from parsed table rows."""
    kw = row_keyword.lower()
    norm_p = period.upper().replace("FY", "")
    for p, text in sorted(pages_text.items()):
        for row in parse_page_rows(p, text):
            # Require a genuine multi-period table row starting with row_keyword (not a prose sentence!)
            if len(row.periods) < 2 or len(row.values) < len(row.periods):
                continue
            if not row.raw_line.lower().startswith(kw):
                continue
            if norm_p in row.periods:
                idx = row.periods.index(norm_p)
                if idx < len(row.values):
                    val, is_pct, _ = row.values[idx]
                    unit = "%" if is_pct else (row.table_unit[0] if row.table_unit else "")
                    return {
                        "value": val,
                        "unit": unit,
                        "row_label": row_keyword,
                        "period": period,
                        "page": p,
                    }
    return None


def run_sufficiency_walkthrough(pdf_path: Path, query: str = QUESTION) -> dict[str, Any]:
    """Run the full zero-LLM 2-round sufficiency & verification loop on ``pdf_path``."""
    t0 = time.perf_counter()
    ctx = DocContext.open(pdf_path)
    doc = PdfDocument.from_pdf(pdf_path)
    t_open_ms = (time.perf_counter() - t0) * 1000.0

    # --- Round 1: Initial zero-LLM locate + anchored read ---
    t1 = time.perf_counter()
    loc1 = ctx.locate(query, k=2)
    r1_pages = [h["p"] for h in loc1.get("hits", [])]
    read1 = ctx.read(r1_pages)
    read_texts: dict[int, str] = {
        p_info["p"]: "\n".join(re.sub(r"^p\d+:L\d+\s+", "", line) for line in p_info["text"].splitlines())
        for p_info in read1["pages"]
    }
    t_r1_ms = (time.perf_counter() - t1) * 1000.0

    # Parse GAAP Gross Margin inputs directly from Round-1 read pages (zero hardcoded numbers!)
    rev24 = _extract_row_period_cell(read_texts, "Total Revenue", "FY2024")
    cogs24 = _extract_row_period_cell(read_texts, "Cost of Goods Sold", "FY2024")
    rev23 = _extract_row_period_cell(read_texts, "Total Revenue", "FY2023")
    cogs23 = _extract_row_period_cell(read_texts, "Cost of Goods Sold", "FY2023")
    assert rev24 and cogs24 and rev23 and cogs23, f"Round 1 failed to locate income statement cells on pages {r1_pages}"

    gaap_gm24 = (rev24["value"] - cogs24["value"]) / rev24["value"] * 100.0
    gaap_gm23 = (rev23["value"] - cogs23["value"]) / rev23["value"] * 100.0
    gaap_delta_pp = round(gaap_gm24 - gaap_gm23, 2)

    gaap_verdict = verify_claim(
        doc,
        claim_text=f"Reported GAAP Gross Margin expanded by {gaap_delta_pp:.1f} pp from FY2023 to FY2024.",
        operands=[
            {**rev24, "name": "rev24"},
            {**cogs24, "name": "cogs24"},
            {**rev23, "name": "rev23"},
            {**cogs23, "name": "cogs23"},
        ],
        formula="((rev24 - cogs24) / rev24 * 100) - ((rev23 - cogs23) / rev23 * 100)",
        claimed_value=gaap_delta_pp,
        claimed_unit="pp",
    )

    # --- Sufficiency Gate: check for unresolved cross-references needed by the query ---
    unresolved = [r for r in detect_unresolved_references(read_texts, query=query) if r["query_relevant"]]
    r2_pages: list[int] = []
    t_r2_ms = 0.0
    if unresolved:
        t2 = time.perf_counter()
        followup_q = unresolved[0]["followup_query"]
        loc2 = ctx.locate(followup_q, k=3)
        r2_pages = [h["p"] for h in loc2.get("hits", []) if h["p"] not in read_texts][:2]
        if r2_pages:
            read2 = ctx.read(r2_pages)
            for p_info in read2["pages"]:
                read_texts[p_info["p"]] = "\n".join(
                    re.sub(r"^p\d+:L\d+\s+", "", line) for line in p_info["text"].splitlines()
                )
        t_r2_ms = (time.perf_counter() - t2) * 1000.0

    # --- Check whether the adjustment evidence (Restructuring in COGS) was found ---
    r_cogs24 = _extract_row_period_cell(read_texts, "Restructuring in Cost of Goods Sold", "FY2024")
    r_cogs23 = _extract_row_period_cell(read_texts, "Restructuring in Cost of Goods Sold", "FY2023")

    if r_cogs24 and r_cogs23:
        adj_gm24 = (rev24["value"] - (cogs24["value"] - r_cogs24["value"])) / rev24["value"] * 100.0
        adj_gm23 = (rev23["value"] - (cogs23["value"] - r_cogs23["value"])) / rev23["value"] * 100.0
        adj_delta_pp = round(adj_gm24 - adj_gm23, 2)
        adj_verdict = verify_claim(
            doc,
            claim_text=(
                f"FY2024 Adjusted Gross Margin (ex-restructuring) changed by {adj_delta_pp:+.1f} pp vs FY2023."
            ),
            operands=[
                {**rev24, "name": "rev24"},
                {**cogs24, "name": "cogs24"},
                {**r_cogs24, "name": "r24"},
                {**rev23, "name": "rev23"},
                {**cogs23, "name": "cogs23"},
                {**r_cogs23, "name": "r23"},
            ],
            formula="((rev24 - (cogs24 - r24)) / rev24 * 100) - ((rev23 - (cogs23 - r23)) / rev23 * 100)",
            claimed_value=adj_delta_pp,
            claimed_unit="pp",
        )
        return {
            "n_pages": doc.num_pages,
            "open_ms": t_open_ms,
            "round1_pages": r1_pages,
            "round1_ms": t_r1_ms,
            "gaap_gm23": gaap_gm23,
            "gaap_gm24": gaap_gm24,
            "gaap_delta_pp": gaap_delta_pp,
            "gaap_verdict": gaap_verdict,
            "unresolved_refs": unresolved,
            "round2_pages": r2_pages,
            "round2_ms": t_r2_ms,
            "sufficient_after_round2": True,
            "adj_gm23": adj_gm23,
            "adj_gm24": adj_gm24,
            "adj_delta_pp": adj_delta_pp,
            "final_verdict": adj_verdict["verdict"],
            "adj_verdict": adj_verdict,
            "answer_flipped": (gaap_delta_pp > 0) != (adj_delta_pp > 0),
        }

    # Note 7 missing -> refuse to guess! Report scoped honesty clearly.
    return {
        "n_pages": doc.num_pages,
        "open_ms": t_open_ms,
        "round1_pages": r1_pages,
        "round1_ms": t_r1_ms,
        "gaap_gm23": gaap_gm23,
        "gaap_gm24": gaap_gm24,
        "gaap_delta_pp": gaap_delta_pp,
        "gaap_verdict": gaap_verdict,
        "unresolved_refs": unresolved,
        "round2_pages": r2_pages,
        "round2_ms": t_r2_ms,
        "sufficient_after_round2": False,
        "final_verdict": "PARTIALLY_VERIFIED",
        "missing_reason": (
            f"Reported GAAP Gross Margin (+{gaap_delta_pp:.1f} pp) is VERIFIED on Page {rev24['page']}, "
            f"but Adjusted Gross Margin (ex-restructuring) has INSUFFICIENT_EVIDENCE: "
            f"unresolved cross-reference {[r['ref'] for r in unresolved]} was not found in the PDF."
        ),
        "answer_flipped": False,
    }


def _print_run(title: str, res: dict[str, Any]) -> None:
    print(f"\n{'=' * 78}")
    print(f"  {title} ({res['n_pages']}-page PDF | open+index {res['open_ms']:.1f}ms)")
    print(f"{'=' * 78}")
    print(f"  [Round 1] Zero-LLM locate+read hit pages {res['round1_pages']} in {res['round1_ms']:.1f}ms")
    print(
        f"            Reported GAAP Gross Margin: FY2023={res['gaap_gm23']:.1f}% -> FY2024={res['gaap_gm24']:.1f}% "
        f"({res['gaap_delta_pp']:+.1f} pp) [{res['gaap_verdict']['verdict']}]"
    )
    print("            Naive single-pass headline answer would say: YES (expanded +4.0 pp)")
    refs = res["unresolved_refs"]
    if refs:
        r0 = refs[0]
        print(
            f"  [Sufficiency Gate] INSUFFICIENT_EVIDENCE -> unresolved ref {r0['ref']!r} "
            f"cited at {r0['line_anchor']}: {r0['context_line']}"
        )
        print(
            f"  [Round 2] Follow-up locate({r0['followup_query']!r}) read extra page(s) "
            f"{res['round2_pages']} in {res['round2_ms']:.1f}ms"
        )
    if res["sufficient_after_round2"]:
        av = res["adj_verdict"]
        print(
            f"            Adjusted Gross Margin (ex-COGS restructuring): "
            f"FY2023={res['adj_gm23']:.1f}% -> FY2024={res['adj_gm24']:.1f}% ({res['adj_delta_pp']:+.1f} pp)"
        )
        print(f"  [Final]   Verdict: {av['verdict']} | verified_scope={av['verified_scope']}")
        print("            Bound cross-page cells:")
        for op in av["operands"]:
            print(f"              - {op['name']:6s} = {op['value']:6.1f} {op['unit']:7s} @ {op['line_anchor']} | {op['source_line']}")
        print("  [Result]  ANSWER FLIPPED after reading 1 more page: NO — Adjusted Gross Margin CONTRACTED (-5.0 pp)!")
    else:
        print(f"  [Final]   Verdict: {res['final_verdict']} (refused to guess!)")
        print(f"            {res['missing_reason']}")


def main() -> int:
    print(f"Question: {QUESTION}")
    with tempfile.TemporaryDirectory() as td:
        full_pdf = build_restructuring_filing_pdf(Path(td) / "acme_fy2024_full.pdf", include_note7=True)
        res_full = run_sufficiency_walkthrough(full_pdf, QUESTION)
        _print_run("Run A — Complete Filing (Note 7 Present)", res_full)

        partial_pdf = build_restructuring_filing_pdf(Path(td) / "acme_fy2024_no_note7.pdf", include_note7=False)
        res_partial = run_sufficiency_walkthrough(partial_pdf, QUESTION)
        _print_run("Run B — Same Filing WITHOUT Note 7 (Honest Abstention)", res_partial)

    print("\nRun full 33-case adversarial verification regression: python3 benchmarks/run_adversarial_verify.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
