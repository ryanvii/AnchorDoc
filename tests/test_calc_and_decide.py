"""Unit tests for deterministic arithmetic, unit safety, and compliance truth table."""
from anchordoc.adjudicate import (
    compare_duration_or_count,
    decide_compliance,
    safe_calc,
    same_quote_with_tax,
)


def test_safe_calc_and_injection_rejection():
    assert abs(safe_calc("(12500 - 8000) / 12500 * 100") - 36.0) < 1e-6
    assert safe_calc("10 / 0") is None
    assert safe_calc("__import__('os').system('echo x')") is None


def test_unit_safe_duration_prevents_48h_vs_48d_bug():
    # 48 hours = 2 days <= 3 days -> ok
    assert compare_duration_or_count(3, "days", 48, "hours", direction="le") == "ok"
    # 72 hours = 3 days > 2 days -> violation
    assert compare_duration_or_count(2, "days", 72, "hours", direction="le") == "violation"
    # Unknown unit MUST NOT silently fall back to required unit
    assert compare_duration_or_count(48, "hours", 48, "weird_unit", direction="le") == "unknown_unit"
    # Warranty lower bound: 24 months >= 1 year -> ok
    assert compare_duration_or_count(1, "year", 24, "months", direction="ge") == "ok"


def test_tax_rate_equivalence():
    assert same_quote_with_tax(100000.0, 113000.0) == 0.13
    assert same_quote_with_tax(100000.0, 150000.0) is None


def test_decide_compliance_truth_table():
    # Hard CiteOrDie gate: unverified quote CANNOT pass
    st, rule = decide_compliance({"offer_kind": "response", "covered": True}, quote_verified=False)
    assert st == "needs_review" and "UNVERIFIED" in rule

    # Mandatory material deferred to post-award -> fail
    st, rule = decide_compliance(
        {"material_required": True, "offer_kind": "promise_later", "covered": False},
        quote_verified=True,
        mandatory=True,
    )
    assert st == "fail" and "MISSING_REQUIRED_MATERIAL" in rule

    # Non-mandatory scored item missing material -> needs_review (never false fatal fail)
    st, _ = decide_compliance(
        {"material_required": True, "offer_kind": "promise_only"},
        quote_verified=True,
        mandatory=False,
    )
    assert st == "needs_review"

    # Verified compliant
    st, rule = decide_compliance(
        {"material_required": True, "offer_kind": "material", "covered": True},
        quote_verified=True,
    )
    assert st == "pass" and rule == "R6_VERIFIED_COMPLIANT"
