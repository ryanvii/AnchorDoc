"""v0.6 unit tests: bounded calc, answer contracts, deterministic composer, provenance, NLI truth table,
SpanCard recall fix, and the reasoning-first / tri-state-gate adjudicators (offline mocks only)."""
from __future__ import annotations

from typing import Any

from anchordoc.adjudicate import FactAdjudicator, safe_calc
from anchordoc.adjudicate.answer_contract import (
    check_contract,
    compose_numeric_answer,
    infer_answer_contract,
    operands_grounded,
    unattested_terms,
)
from anchordoc.adjudicate.engine import decide_nli
from anchordoc.docbase import PdfDocument
from anchordoc.index.spancard import SpanCardIndex
from anchordoc.runtime.text import rrf, tokenize


# ---------------------------------------------------------------- calc
def test_safe_calc_pow_unicode_percent_and_bounds():
    assert abs(safe_calc("(150/100) ** (1/3) - 1") - 0.144714) < 1e-5  # CAGR
    assert abs(safe_calc("10 \u2212 4 \u00d7 2") - 2.0) < 1e-9  # unicode minus / times
    assert abs(safe_calc("2^10") - 1024.0) < 1e-9
    assert abs(safe_calc("65% - 53%") - 12.0) < 1e-9  # pure percentage-point arithmetic
    assert abs(safe_calc("12.5% * 400") - 50.0) < 1e-9  # mixed -> percent means /100
    assert safe_calc("9 ** 9 ** 9") is None  # power bomb rejected
    assert safe_calc("(-8) ** 0.5") is None  # complex rejected
    long_expr = " + ".join(["1"] * 120)
    assert safe_calc(long_expr) == 120.0  # old 160-char cap lifted (bounded at 600)
    assert safe_calc("__import__('os')") is None


# ---------------------------------------------------------------- contracts
def test_infer_answer_contract_kinds_scale_and_rounding():
    c = infer_answer_contract("How much (in USD billions) did the company pay in dividends for FY2020?")
    assert c.scale_label == "billion" and c.scale_factor == 1e9
    c = infer_answer_contract("What is the 3 year average margin? Answer in percents and round to one decimal place.")
    assert c.decimals == 1
    assert infer_answer_contract("How many adults rated the policy as poor?").kind == "count"
    assert infer_answer_contract("How many percent of tweets are in English?").kind == "percent"
    assert infer_answer_contract("Does the company have positive working capital?").kind == "yesno"
    assert infer_answer_contract("When did the number of tweets exceed 200,000?").kind == "date"


def test_check_contract_flags_only_clear_shape_errors():
    cnt = infer_answer_contract("How many adults rated the policy as poor?")
    assert "COUNT_EXPECTED_GOT_PERCENT" in check_contract(cnt, "36%")
    assert check_contract(cnt, "541 adults (36% of 1,503)") == []
    bil = infer_answer_contract("How much (in USD billions) were dividends?")
    assert check_contract(bil, "389", "USD millions") == ["SCALE_EXPECTED_BILLION"]
    assert check_contract(bil, "$0.39 billion") == []


# ---------------------------------------------------------------- composer
def test_composer_replaces_wrong_llm_number_with_python_value():
    q = "What is the FY2017 days payable outstanding? Round your answer to two decimal places."
    c = infer_answer_contract(q)
    f = "365*((25309+34616)/2)/(111934+(16047-11461))"
    ans, tag = compose_numeric_answer("81.63", safe_calc(f), f, c, "days")
    assert tag == "CALC_REPLACED" and ans.startswith("93.86")
    f2 = "((2112/14082)+(509/11303)+(1306/11503))/3*100"
    c2 = infer_answer_contract("What is the 3 year average operating margin? Answer in units of percents and round to one decimal place.")
    ans2, tag2 = compose_numeric_answer("10.9", safe_calc(f2), f2, c2, "percents")
    assert tag2 == "CALC_REPLACED" and ans2.startswith("10.3")


def test_composer_never_overwrites_stance_answers():
    c = infer_answer_contract("Does the company have a healthy quick ratio?")
    f = "(2605 + 24506) / 50171"
    ans, tag = compose_numeric_answer("No. The quick ratio is about 0.54, which is low.", safe_calc(f), f, c, "x")
    assert ans.startswith("No.") and tag == "CALC_ANNOTATED_STANCE"
    # Conflicting stance keeps prose untouched (no "(Python-verified value: ...)" injected into body)
    ans2, tag2 = compose_numeric_answer("Yes, liquidity is fine.", safe_calc(f), f, c, "x")
    assert ans2.startswith("Yes, liquidity is fine. [Python-verified:")
    assert tag2 == "CALC_CONFLICT_STANCE" and "Python-verified value" not in ans2
    # Yes/No contract NEVER replaces even when the LLM starts its sentence with a number
    c_yn = infer_answer_contract("Are the company's FY2022 financials that of a high growth company?")
    ans_yn, tag_yn = compose_numeric_answer(
        "1.25% revenue growth — No, this is not a high growth company.", 1.2455, "(94943/93775-1)*100", c_yn, "%"
    )
    assert ans_yn.startswith("1.25% revenue growth — No") and tag_yn == "CALC_ANNOTATED_STANCE"
    # Ungrounded operands never overwrite an LLM answer
    ans_ug, tag_ug = compose_numeric_answer("1.57", 1.7661, "9999/5662", infer_answer_contract("What is the quick ratio?"), "", operand_grounded=False)
    assert ans_ug.startswith("1.57 [Python-verified:") and tag_ug == "CALC_UNGROUNDED_KEPT"
    # sign expressed in words still counts as consistent
    c3 = infer_answer_contract("Did total debt increase?")
    ans3, tag3 = compose_numeric_answer("No. Debt fell by $2,544 million.", -2544.0, "47237-49781", c3, "")
    assert tag3 == "CALC_ANNOTATED_STANCE"


def test_composer_rejects_percent_form_for_decimal_ratio_question():
    q = "What is the FY2017 return on assets (ROA)? Round your answer to two decimal places."
    c = infer_answer_contract(q)
    f = "1283 / ((87896 + 87270) / 2)"
    ans, tag = compose_numeric_answer("1.47%", safe_calc(f), f, c, "%")
    assert tag == "CALC_REPLACED" and ans.startswith("0.01 [Python-verified:")
    # Explicit ratio question with LLM answering in %
    c_r = infer_answer_contract("What is the current ratio? Round to two decimal places.")
    ans_r, tag_r = compose_numeric_answer("54%", 0.5403, "27111/50171", c_r, "%")
    assert tag_r == "CALC_REPLACED" and ans_r.startswith("0.54 [Python-verified:")
    # When formula itself multiplies by 100, percentage answer remains consistent
    c_p = infer_answer_contract("What is the FY2022 margin? Round to one decimal place.")
    ans_p, tag_p = compose_numeric_answer("30.8%", 30.7963, "308/1000*100", c_p, "%")
    assert tag_p == "CALC_CONSISTENT" and ans_p.startswith("30.8%")


def test_count_contract_rounds_to_integer():
    c = infer_answer_contract("How many adults rated it as poor?")
    ans, tag = compose_numeric_answer("", safe_calc("0.36 * 1503"), "0.36 * 1503", c, "")
    assert ans.startswith("541 ") and tag == "CALC_REPLACED"


# ---------------------------------------------------------------- provenance
def test_operands_grounded_with_scale_and_rounding():
    src = "Revenue $ 14,082 million ... Operating income 2,112 ... Cash 4,835 ... Liabilities 6,369"
    ok, miss = operands_grounded("2112/14082*100", src)
    assert ok and miss == []
    # Thousands commas inside formula must not split 4,835 into 4 and 835
    ok_comma, miss_comma = operands_grounded("4,835 / 6,369", src)
    assert ok_comma and miss_comma == []
    ok2, miss2 = operands_grounded("2112/99999*100", src)
    assert not ok2 and miss2 == ["99999"]
    ok3, _ = operands_grounded("14.082/2", "Revenue 14,082")  # thousand scaling
    assert ok3


def test_unattested_terms_is_conservative():
    hay = ["Germanwings and Lufthansa tweets in 2015", "[VISION:CHART] Airbus followers"]
    assert unattested_terms("How many Tweets are attributed to Boeing?", hay) == ["Boeing"]
    assert unattested_terms("How many tweets did Germanwing post in 2015?", hay) == []  # fuzzy typo tolerance
    assert unattested_terms("What happened in 2022?", hay) == ["2022"]
    assert unattested_terms("List the years, format like ['2006', '2007']", hay) == []
    assert unattested_terms("What was FY22 revenue for 2022?", ["revenue fy22 grew"]) == []


# ---------------------------------------------------------------- NLI truth table
def test_decide_nli_truth_table():
    # TOPIC_ABSENT -> high-confidence neutral
    assert decide_nli({"topic_clauses": [], "stance": "silent"}) == ("neutral", "NLI_TOPIC_ABSENT", "high")
    # Dual-fact agreement: existential needs permitted_in_some_case AND stance=supports
    ex_ok = {"topic_clauses": ["s0003"], "hyp_quantifier": "existential", "stance": "supports", "permitted_in_some_case": True}
    assert decide_nli(ex_ok, True) == ("entailment", "NLI_EXISTENTIAL_WITNESS", "high")
    # Conflicting facts (permitted=True while stance=silent) -> abstain neutral + low confidence
    ex_conflict = {"topic_clauses": ["s0003"], "hyp_quantifier": "existential", "stance": "silent", "permitted_in_some_case": True}
    assert decide_nli(ex_conflict, True) == ("neutral", "NLI_EXISTENTIAL_CONFLICT", "low")
    excl = {"topic_clauses": ["s0003"], "hyp_quantifier": "existential", "stance": "opposes", "permitted_in_some_case": "false"}
    assert decide_nli(excl, True) == ("contradiction", "NLI_EXISTENTIAL_EXCLUDED", "high")
    # Universal dual-fact agreement: counterexample + opposes -> contradiction; counterexample + silent -> conflict
    uni_ok = {"topic_clauses": ["s0002"], "hyp_quantifier": "universal", "stance": "opposes", "counterexample": True}
    assert decide_nli(uni_ok, True) == ("contradiction", "NLI_UNIVERSAL_COUNTEREXAMPLE", "high")
    uni_conflict = {"topic_clauses": ["s0002"], "hyp_quantifier": "universal", "stance": "silent", "counterexample": True}
    assert decide_nli(uni_conflict, True) == ("neutral", "NLI_UNIVERSAL_CONFLICT", "low")
    # Topic present yet silent -> low confidence neutral
    assert decide_nli({"topic_clauses": ["s1"], "hyp_quantifier": "plain", "stance": "silent"}, True) == ("neutral", "NLI_TOPIC_SILENT", "low")
    # Plain stance + quote_verified tiers
    assert decide_nli({"topic_clauses": ["s1"], "hyp_quantifier": "plain", "stance": "supports"}, True) == ("entailment", "NLI_STANCE_SUPPORTS", "high")
    assert decide_nli({"topic_clauses": ["s1"], "stance": "supports"}, False)[2] == "medium"


# ---------------------------------------------------------------- SpanCard fix
def test_spancard_keeps_all_topk_hits_before_neighbors():
    paras = [f"Clause {i}. Generic filler text about delivery schedules and invoices number {i}." for i in range(40)]
    for i in (3, 11, 19, 27, 35):
        paras[i] = f"Clause {i}. The receiving party shall return confidential information upon request {i}."
    idx = SpanCardIndex("\n\n".join(paras))
    q = "return confidential information upon request"
    qt = tokenize(q)
    a, b = idx.bm_raw.get_scores(qt), idx.bm_card.get_scores(qt)
    fused = rrf(sorted(range(len(idx.spans)), key=lambda i: -a[i])[:20], sorted(range(len(idx.spans)), key=lambda i: -b[i])[:20], k=60)[:7]
    got = {sp.sid - 1 for sp in idx.retrieve(q, top_k=7)}
    assert set(fused) <= got


# ---------------------------------------------------------------- adjudicators (offline mocks)
class _Mock:
    model = "mock"

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def snapshot_stats(self) -> dict[str, Any]:
        return {"calls": len(self.calls)}

    def chat_json(self, purpose, system, user, max_tokens=700, images_b64=None):
        self.calls.append((purpose, len(images_b64 or [])))
        r = self.responses.get(purpose, {})
        return (r(user) if callable(r) else r), "{}"


def test_answer_numerical_replaces_number_and_records_provenance():
    doc = PdfDocument.from_pages(["Cover", "Statement\nRevenues $1,200 Cost of sales $720\nGross profit $480"])
    llm = _Mock({
        "anchordoc_num_qa_v6": {
            "evidence": [{"page": 2, "quote": "Revenues $1,200 Cost of sales $720"}],
            "operands": [], "formula": "(1200 - 720) / 1200 * 100", "reasoning": "margin",
            "can_answer": True, "unit": "%", "final_answer": "38.5%", "cited_pages": [2],
        }
    })
    v = FactAdjudicator(llm).answer_numerical(doc, "What is the gross margin percentage?", [2])
    assert v.status == "answered" and v.answer.startswith("40")
    assert v.meta["composition"] == "CALC_REPLACED" and v.meta["operand_grounded"] is True
    assert v.quote_verified is True and "OPERANDS_GROUNDED" in v.rule_trace


def test_answer_numerical_single_repair_on_contract_violation():
    doc = PdfDocument.from_pages(["Survey of 1,503 adults. 36% rated it poor."])
    llm = _Mock({
        "anchordoc_num_qa_v6": {"evidence": [], "formula": "", "can_answer": True, "final_answer": "36%", "cited_pages": [1]},
        "anchordoc_num_repair": {"formula": "0.36 * 1503", "unit": "adults", "final_answer": "541"},
    })
    v = FactAdjudicator(llm).answer_numerical(doc, "How many adults rated it as poor?", [1])
    assert [c for c, _ in llm.calls].count("anchordoc_num_repair") == 1
    assert v.meta["repaired"] is True and v.answer.startswith("541")


class _RenderDoc(PdfDocument):
    """PdfDocument test double that 'renders' pages and tiles without a real PDF."""

    def render_page_png_b64(self, page_num: int, scale: float = 1.5) -> str:
        return f"img{page_num}"

    def render_page_tiles_b64(self, page_num: int, grid: int = 2, overlap: float = 0.1, scale: float = 2.0) -> list[str]:
        return [f"tile{page_num}_{i}" for i in range(grid * grid)]


def _mm_doc() -> _RenderDoc:
    return _RenderDoc(path=None, pages=["", "Germanwings English tweets 10 German tweets 14", "", "", "", ""])


def test_answer_multimodal_derivable_is_answered_with_zoom_and_all_topk():
    llm = _Mock({
        "anchordoc_mm_qa_v6": {
            "observations": [{"page": 2, "detail": "English tweets 10 German tweets 14"}],
            "evidence_status": "derivable", "entity_time_match": True,
            "derivation": "10/(10+14)", "formula": "10/(10+14)*100", "reasoning": "share",
            "final_answer": "41.67%", "cited_pages": [2],
        }
    })
    v = FactAdjudicator(llm).answer_multimodal(_mm_doc(), "How many percent of Germanwings tweets are in English?", [2, 1, 3, 4, 5])
    assert v.status == "answered" and v.answer.startswith("41.67")
    assert v.meta["rendered_pages"] == [2, 1, 3, 4, 5]  # all top-5 rendered (was 4)
    assert v.meta["zoom_tiles"] == 4 and llm.calls[0][1] == 9
    assert v.meta["grounding"] == "text_verified" and v.quote_verified is True


def test_answer_multimodal_gates_absent_mismatch_and_unattested():
    adj_absent = FactAdjudicator(_Mock({"anchordoc_mm_qa_v6": {"evidence_status": "absent", "final_answer": "Not answerable"}}))
    assert adj_absent.answer_multimodal(_mm_doc(), "What is the revenue?", [2]).status == "abstain"
    adj_mis = FactAdjudicator(_Mock({"anchordoc_mm_qa_v6": {"evidence_status": "stated", "entity_time_match": False, "final_answer": "5"}}))
    assert "MM_ENTITY_TIME_MISMATCH" in adj_mis.answer_multimodal(_mm_doc(), "How many tweets?", [2]).rule_trace
    adj_un = FactAdjudicator(_Mock({"anchordoc_mm_qa_v6": {"evidence_status": "stated", "entity_time_match": True, "final_answer": "5"}}))
    v = adj_un.answer_multimodal(_mm_doc(), "How many Tweets are attributed to Boeing?", [2])
    assert v.status == "abstain" and v.meta["unattested_terms"] == ["Boeing"]


def test_audit_prose_clause_two_stage_backward_compatible():
    text = "3. RETURN. The Recipient may keep one archival copy of Confidential Information for legal purposes."
    llm = _Mock({
        "anchordoc_prose_nli_v6": {
            "topic_clauses": ["s0001"],
            "verbatim_quote": "The Recipient may keep one archival copy of Confidential Information",
            "analysis": "archival copy permitted", "hyp_quantifier": "existential", "stance": "supports",
            "permitted_in_some_case": True, "counterexample": False,
        }
    })
    idx = SpanCardIndex(text)
    v = FactAdjudicator(llm).audit_prose_clause(text, "Receiving Party may retain some Confidential Information.", idx.spans)
    assert v.answer == "entailment" and v.status == "pass" and v.facts["confidence"] == "high"
    assert v.rule_trace.startswith("NLI_EXISTENTIAL_WITNESS")
