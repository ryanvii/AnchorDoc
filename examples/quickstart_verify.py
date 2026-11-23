"""10-Second Zero-API-Key Quickstart Demo: Verify a Correct Filing Claim, Then Catch a Period Swap!

Run from any clean clone (0 external downloads, 0 LLM API keys):
    python3 examples/quickstart_verify.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchordoc import DocContext, PdfDocument, verify_claim  # noqa: E402
from benchmarks.run_adversarial_verify import build_sample_filing_pdf  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        pdf_path = build_sample_filing_pdf(Path(td) / "acme_fy2024_filing.pdf")
        ctx = DocContext.open(pdf_path)
        doc = PdfDocument.from_pdf(pdf_path)

        print("=========================================================================")
        print("  AnchorDoc 10-Second Zero-API-Key Walkthrough (3-Page Sample 10-K PDF)")
        print("=========================================================================")

        # 0. Zero-LLM cross-page outline & locate
        outline = ctx.outline()
        loc = ctx.locate("Gross Profit FY2024 FY2023 Total Revenue", k=2)
        top_pages = [(h["p"], h["section"]) for h in loc.get("hits", [])]
        print(f"\n[Step 0] Zero-LLM Outline ({outline['n_pages']}p) & Locator found target pages in <5ms:")
        for p, sec in top_pages:
            print(f"  -> page {p}: {sec}")

        # 1. Verify the TRUE FY2024 Gross Profit claim
        print("\n[Run 1] Verify TRUE claim: FY2024 Gross Profit = rev - cogs = 1,200 - 720 = 480 ($M)")
        ok = verify_claim(
            doc,
            claim_text="FY2024 Gross Profit is 480 million.",
            operands=[
                {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
                {"name": "cogs", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
            ],
            formula="rev - cogs",
            claimed_value=480,
            claimed_unit="million",
        )
        print(f"  Verdict        : {ok['verdict']} (verified_scope={ok['verified_scope']})")
        print(f"  AST Recompute  : {ok['formula']} = {ok['calc_value']} million")
        for op in ok["operands"]:
            print(f"  Bound Cell     : {op['name']}={op['value']} -> {op['line_anchor']} | {op['source_line']}")

        # 2. Same filing, same question — swap ONE number to FY2023's Gross Profit (378)
        print("\n[Run 2] Catch a subtle YEAR-COLUMN SWAP: same page, AI grabs 378 (FY2023) as FY2024")
        bad = verify_claim(
            doc,
            claim_text="FY2024 Gross Margin is 31.5% (378 / 1200).",
            operands=[
                {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
                {"name": "gp", "value": 378, "unit": "million", "row_label": "Gross Profit", "period": "FY2024", "page": 2},
            ],
            formula="gp / rev * 100",
            claimed_value=31.5,
            claimed_unit="%",
        )
        print(f"  Verdict        : {bad['verdict']} | error_codes={bad['error_codes']}")
        print(f"  Per-Layer Check: {bad['checks']}")
        for d in bad["diagnostics"]:
            print(f"  Diagnosis      : {d}")
        print("\nDone! Run full 31-case synthetic regression: python3 benchmarks/run_adversarial_verify.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
