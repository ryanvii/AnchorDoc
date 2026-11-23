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
