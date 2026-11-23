"""Adversarial verification regression suite (v0.9.0): sign flips, period/unit/row/pp traps."""
from __future__ import annotations

import json
from pathlib import Path

from anchordoc import DocContext, PdfDocument, verify_claim, verify_quote
from anchordoc.adjudicate.answer_contract import operands_grounded
from anchordoc.cli import main as cli_main
from anchordoc.mcp_server import AnchorDocMCPApp
from benchmarks.run_adversarial_verify import build_sample_filing_pdf, run_benchmark


def test_verify_quote_rejects_sign_flip_even_when_positive_exists_on_same_page():
    page = (
        "Total revenue growth was 12% year-over-year. "
        "However, EMEA Operating margin was -12% due to restructuring."
    )
    assert verify_quote("EMEA Operating margin was -12% due to restructuring.", page)[0] is True
    assert verify_quote("EMEA Operating margin was 12% due to restructuring.", page)[0] is False


def test_verify_quote_handles_accounting_parenthetical_negatives():
    page = "Consolidated net loss in Q4 was $(145) million; margin declined to (8.4%)."
    assert verify_quote("Consolidated net loss in Q4 was -$145 million", page)[0] is True
    assert verify_quote("Consolidated net loss in Q4 was $145 million", page)[0] is False
    assert verify_quote("margin declined to -8.4%", page)[0] is True
    assert verify_quote("margin declined to 8.4%", page)[0] is False


def test_operands_grounded_enforces_sign_and_strict_scale():
    src = "Operating margin was -12% and Restructuring Expense was $(145) million."
    ok_pos, missing_pos = operands_grounded("12 + 145", src)
    assert ok_pos is False
    assert "12" in missing_pos and "145" in missing_pos

    ok_neg, missing_neg = operands_grounded("-12 + -145", src)
    assert ok_neg is True and missing_neg == []

    # strict_scale rejects 1000x implicit scaling
    ok_loose, _ = operands_grounded("145000", "Revenue was 145 million", strict_scale=False)
    ok_strict, _ = operands_grounded("145000", "Revenue was 145 million", strict_scale=True)
    assert ok_loose is True
    assert ok_strict is False


def test_adversarial_mini_bench_zero_false_pass_and_zero_false_alarm():
    res = run_benchmark()
    nv = res["naive_verifier"]
    ad = res["anchordoc_v09"]
    # Naive check misses almost all adversarial traps!
    assert nv["false_pass_count"] >= 10
    # AnchorDoc v0.9.0 catches 100% of traps with 0% false alarms!
    assert ad["trap_catch_rate_%"] == 100.0
    assert ad["false_pass_count"] == 0
    assert ad["false_alarm_count"] == 0
    assert ad["exact_error_code_match"] == f"{res['adversarial_traps']}/{res['adversarial_traps']}"


def test_cli_verify_claim_and_mcp_doc_verify_end_to_end(tmp_path: Path, capsys):
    pdf = build_sample_filing_pdf(tmp_path / "filing.pdf")
    ops = json.dumps([
        {"name": "FY2024 Gross Profit", "value": 378, "unit": "million", "row_label": "Gross Profit", "period": "FY2024", "page": 2}
    ])
    rc = cli_main(["verify-claim", str(pdf), "--operands-json", ops])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["verified"] is False
    assert "PERIOD_MISMATCH" in out["error_codes"]

    # MCP doc_verify also supports structured external claim verification
    app = AnchorDocMCPApp(roots=[tmp_path])
    mcp_res = app.call_tool("doc_verify", {
        "path": str(pdf),
        "claim_text": "Gross margin increased by 4.0% from FY2023 to FY2024.",
        "operands": [
            {"name": "FY2024 GM", "value": 40.0, "unit": "%", "row_label": "Gross Margin", "period": "FY2024", "page": 2},
            {"name": "FY2023 GM", "value": 36.0, "unit": "%", "row_label": "Gross Margin", "period": "FY2023", "page": 2},
        ],
        "formula": "40.0 - 36.0",
        "claimed_value": 4.0,
        "claimed_unit": "%",
    })
    assert mcp_res["verified"] is False
    assert "PP_VS_PCT_CONFUSION" in mcp_res["error_codes"]

    # Valid claim via DocContext.verify returns VERIFIED
    ctx = DocContext.open(pdf)
    ok_res = ctx.verify(
        operands=[
            {"name": "FY2024 Revenue", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "FY2024 COGS", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        formula="1200 - 720",
        claimed_value=480,
        claimed_unit="million",
    )
    assert ok_res["verified"] is True
    assert ok_res["calc_value"] == 480.0


def test_scoped_honesty_blocks_evidence_laundering_and_irrelevant_quote(tmp_path: Path):
    """Regression tests for exact red-team cases:
    1. Valid operands [1200, 720] + sneaky formula `1200 - 700 = 500` MUST FAIL (`FORMULA_OPERAND_UNBOUND`).
    2. False numeric claim `Gross profit is 500` + true unrelated covenant quote MUST NOT return VERIFIED
       (returns `PARTIALLY_VERIFIED` with quotes=verified, claim_alignment=unverified).
    3. Symbolic formula `rev - cogs = 480` bound directly to verified cell operands returns `VERIFIED`
       with explicit `checks` and `verified_scope`.
    """
    pdf = build_sample_filing_pdf(tmp_path / "filing.pdf")
    doc = PdfDocument.from_pdf(pdf)

    # Case 1: Evidence laundering attack (valid operands + wrong literal in formula)
    launder = verify_claim(
        doc,
        operands=[
            {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "cogs", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        formula="1200 - 700",
        claimed_value=500,
        claimed_unit="million",
    )
    assert launder["verdict"] == "FAILED"
    assert launder["verified"] is False
    assert "FORMULA_OPERAND_UNBOUND" in launder["error_codes"]
    assert launder["checks"]["operands"] == "verified"
    assert launder["checks"]["computation"] == "failed"

    # Case 2: Unrelated true quote attached to unverified numeric claim
    irrel = verify_claim(
        doc,
        claim_text="Gross profit is 500 million.",
        quotes=[{"text": "Debt-to-Asset Ratio shall not exceed 65.0% at fiscal year-end.", "page": 3}],
    )
    assert irrel["verdict"] == "PARTIALLY_VERIFIED"
    assert irrel["verified"] is False
    assert irrel["checks"]["quotes"] == "verified"
    assert irrel["checks"]["operands"] == "not_checked"
    assert irrel["checks"]["computation"] == "not_checked"
    assert irrel["checks"]["claim_alignment"] == "unverified"
    assert "CLAIM_UNSUPPORTED_BY_EVIDENCE" in irrel["warning_codes"]

    # Case 3: Symbolic formula referencing bound operands directly
    sym = verify_claim(
        doc,
        claim_text="FY2024 Gross profit is 480 million.",
        operands=[
            {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "cogs", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        formula="rev - cogs",
        claimed_value=480,
        claimed_unit="million",
    )
    assert sym["verdict"] == "VERIFIED"
    assert sym["verified"] is True
    assert sym["calc_value"] == 480.0
    assert sym["checks"]["operands"] == "verified"
    assert sym["checks"]["computation"] == "verified"
    assert sym["checks"]["claim_alignment"] == "verified"
    assert set(sym["verified_scope"]) == {"operands", "computation", "claim_alignment"}


def test_v092_redteam_mixed_formula_metric_swap_and_claim_period(tmp_path: Path):
    """Exact regression tests for the 3 red-team escape cases in v0.9.2:
    1. Mixed formula `rev - 700 = 500` with bound `rev=1200, cogs=720` -> FAILED (`FORMULA_OPERAND_UNBOUND`).
    2. `FY2024 Gross Margin is 65.0%` + Debt Covenant `<= 65.0%` quote -> PARTIALLY_VERIFIED (NOT VERIFIED!).
    3. FY2024 bound math `rev - cogs = 480` + claim text saying `FY2023` -> FAILED (`CLAIM_PERIOD_MISMATCH`).
    """
    pdf = build_sample_filing_pdf(tmp_path / "filing.pdf")
    doc = PdfDocument.from_pdf(pdf)

    # 1. Mixed symbol + unbound literal
    mixed = verify_claim(
        doc,
        operands=[
            {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "cogs", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        formula="rev - 700",
        claimed_value=500,
        claimed_unit="million",
    )
    assert mixed["verdict"] == "FAILED"
    assert mixed["verified"] is False
    assert "FORMULA_OPERAND_UNBOUND" in mixed["error_codes"]

    # 2. Same number (65.0%) but different metric/meaning in quote-only mode -> NEVER VERIFIED
    metric_swap = verify_claim(
        doc,
        claim_text="FY2024 Gross Margin is 65.0%.",
        quotes=[{"text": "Debt-to-Asset Ratio shall not exceed 65.0% at fiscal year-end.", "page": 3}],
    )
    assert metric_swap["verdict"] == "PARTIALLY_VERIFIED"
    assert metric_swap["verified"] is False
    assert metric_swap["checks"]["quotes"] == "verified"
    assert metric_swap["checks"]["claim_alignment"] == "unverified"

    # 3. FY2024 bound operands + FY2023 stated in claim_text -> CLAIM_PERIOD_MISMATCH
    year_swap = verify_claim(
        doc,
        claim_text="FY2023 Gross Profit is 480 million.",
        operands=[
            {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
            {"name": "cogs", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
        ],
        formula="rev - cogs",
        claimed_value=480,
        claimed_unit="million",
    )
    assert year_swap["verdict"] == "FAILED"
    assert year_swap["verified"] is False
    assert "CLAIM_PERIOD_MISMATCH" in year_swap["error_codes"]
    assert year_swap["checks"]["claim_alignment"] == "failed"


def test_v093_claim_metric_mismatch_preserves_verified_computation(tmp_path: Path):
    """Exact regression test for v0.9.3 metric-name swap:
    rev - cogs = 480 (Gross Profit) claimed as 'Net Income is 480' (actual Net Income = 125)
    MUST keep computation='verified' while failing claim_alignment with CLAIM_METRIC_MISMATCH.
    """
    pdf = build_sample_filing_pdf(tmp_path / "filing.pdf")
    doc = PdfDocument.from_pdf(pdf)

    for claim in ("FY2024 Net Income is 480 million.", "FY2024 净利润为 480 百万。"):
        res = verify_claim(
            doc,
            claim_text=claim,
            operands=[
                {"name": "rev", "value": 1200, "unit": "million", "row_label": "Total Revenue", "period": "FY2024", "page": 2},
                {"name": "cogs", "value": 720, "unit": "million", "row_label": "Cost of Goods Sold", "period": "FY2024", "page": 2},
            ],
            formula="rev - cogs",
            claimed_value=480,
            claimed_unit="million",
        )
        assert res["verdict"] == "FAILED"
        assert res["verified"] is False
        assert "CLAIM_METRIC_MISMATCH" in res["error_codes"]
        assert res["checks"]["operands"] == "verified"
        assert res["checks"]["computation"] == "verified"
        assert res["checks"]["claim_alignment"] == "failed"


def test_v093_evidence_sufficiency_walkthrough_flips_answer_and_abstains_without_note7(tmp_path: Path):
    """Verify the homepage restructuring-footnote sufficiency demo runs end-to-end with 0 API keys:
    - With Note 7 on Page 6: Round 1 hits Income Statement (+4.0 pp GAAP), detects missing Note 7,
      Round 2 locates Page 6, recomputes Adjusted Gross Margin across Page 2 + Page 6 (-5.0 pp) -> VERIFIED.
    - Without Note 7: Round 2 finds no Note 7 -> refuses to guess -> PARTIALLY_VERIFIED.
    """
    from examples.quickstart_verify import build_restructuring_filing_pdf, run_sufficiency_walkthrough

    full_pdf = build_restructuring_filing_pdf(tmp_path / "full.pdf", include_note7=True)
    res_full = run_sufficiency_walkthrough(full_pdf)
    assert 2 in res_full["round1_pages"]
    assert 6 not in res_full["round1_pages"]
    assert any(r["ref"].lower() == "note 7" for r in res_full["unresolved_refs"])
    assert 6 in res_full["round2_pages"]
    assert res_full["sufficient_after_round2"] is True
    assert res_full["gaap_delta_pp"] == 4.0
    assert res_full["adj_delta_pp"] == -5.0
    assert res_full["answer_flipped"] is True
    assert res_full["final_verdict"] == "VERIFIED"

    no_note7_pdf = build_restructuring_filing_pdf(tmp_path / "no_note7.pdf", include_note7=False)
    res_partial = run_sufficiency_walkthrough(no_note7_pdf)
    assert res_partial["gaap_verdict"]["verdict"] == "VERIFIED"
    assert res_partial["sufficient_after_round2"] is False
    assert res_partial["final_verdict"] == "PARTIALLY_VERIFIED"
    assert res_partial["failure_stage"] == "REF_NOT_FOUND"
    assert "was not found in the PDF" in res_partial["missing_reason"]


def test_v093_quickstart_dynamic_output_and_parsed_row_failure_stage(tmp_path: Path, capsys):
    """Ensure quickstart_verify.py reports actual delta/direction dynamically and distinguishes
    'Note 7 found on Page 6, but allocation row not parsed' from 'Note 7 missing from PDF'.
    """
    from examples.quickstart_verify import _print_run, build_restructuring_filing_pdf, run_sufficiency_walkthrough

    # 1. Dynamic output when adjusted margin actually EXPANDS (+2.0 pp, answer_flipped=False)
    full_pdf = build_restructuring_filing_pdf(tmp_path / "full.pdf", include_note7=True)
    res_full = run_sufficiency_walkthrough(full_pdf)
    mutated_res = dict(res_full, adj_gm24=48.0, adj_delta_pp=2.0, answer_flipped=False)
    _print_run("Mutated Expansion", mutated_res)
    out = capsys.readouterr().out
    assert "ANSWER CONFIRMED" in out
    assert "+2.0 pp" in out
    assert "ANSWER FLIPPED" not in out
    assert "-5.0 pp" not in out

    # 2. Note 7 present on Page 6, but row renamed to 'Restructuring in Cost of Sales'
    renamed_pdf = tmp_path / "renamed_row.pdf"
    raw = full_pdf.read_bytes()
    assert b"Restructuring in Cost of Goods Sold" in raw
    # Preserve PDF byte length / xref offsets cleanly by building with identical-length replacement
    # ("Restructuring in Cost of Goods Sold" = 35 chars -> "Restructuring in Cost of Sales Item" = 35 chars)
    renamed_pdf.write_bytes(raw.replace(b"Restructuring in Cost of Goods Sold", b"Restructuring in Cost of Sales Item"))
    res_renamed = run_sufficiency_walkthrough(renamed_pdf)
    assert res_renamed["final_verdict"] == "PARTIALLY_VERIFIED"
    assert res_renamed["failure_stage"] == "ALLOCATION_ROW_NOT_PARSED"
    assert "was found on page(s) [6]" in res_renamed["missing_reason"]
    assert "was not found in the PDF" not in res_renamed["missing_reason"]




