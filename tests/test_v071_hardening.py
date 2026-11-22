"""Regression unit tests for AnchorDoc v0.7.1 P0 & P1 hardening fixes."""
from __future__ import annotations

import json
import sqlite3
import types
from pathlib import Path

import pytest

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.context import DocContext
from anchordoc.docbase import PdfDocument
from anchordoc.mcp_server import AnchorDocMCPApp, split_mcp_payload
from anchordoc.runtime.llm import CachedLLM
from anchordoc.store.bundle import BundleStore, TextLayer, _is_corruption_error


def test_p0_1_transient_operational_error_never_quarantines_healthy_bundle(tmp_path, monkeypatch):
    store = BundleStore(tmp_path / "bundles")
    sha = "e" * 64
    store.save_text(sha, "ext1", TextLayer(pages=["Healthy page content"]))
    db_path = store.bundle_path(sha)
    assert db_path.is_file()

    assert _is_corruption_error(sqlite3.OperationalError("database is locked")) is False
    assert _is_corruption_error(sqlite3.OperationalError("unable to open database file")) is False
    assert _is_corruption_error(sqlite3.DatabaseError("file is not a database")) is True
    assert _is_corruption_error(sqlite3.OperationalError("database disk image is malformed")) is True

    # Simulate transient lock on open -> must raise, MUST NOT rename healthy DB to .corrupt!
    monkeypatch.setattr(store, "_init_conn", lambda p: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")))
    with pytest.raises(sqlite3.OperationalError):
        store.load_text(sha, "ext1")
    assert db_path.is_file()
    assert not db_path.with_suffix(".sqlite.corrupt").exists()


def test_p0_3_mcp_splits_images_out_of_json_text_payload():
    raw = {
        "doc_id": "abc",
        "pages": [
            {"p": 1, "text": "p1:L01 hello", "images_b64": ["FAKE_B64_PNG_1", "FAKE_B64_PNG_2"]},
            {"p": 2, "text": "p2:L01 world"},
        ],
    }
    clean, imgs = split_mcp_payload(raw)
    assert imgs == ["FAKE_B64_PNG_1", "FAKE_B64_PNG_2"]
    assert "images_b64" not in clean["pages"][0]
    assert clean["pages"][0]["images_returned"] == 2
    assert "FAKE_B64_PNG_1" not in json.dumps(clean)


def test_p0_2_mcp_tools_have_readonly_annotations_and_string_quotes_schema(tmp_path):
    app = AnchorDocMCPApp(roots=[tmp_path])
    tools = app.list_tools()
    assert len(tools) == 4
    for t in tools:
        assert t["annotations"]["readOnlyHint"] is True
        assert t["annotations"]["destructiveHint"] is False
    verify_tool = next(t for t in tools if t["name"] == "doc_verify")
    assert "oneOf" in verify_tool["inputSchema"]["properties"]["quotes"]["items"]


def test_p0_4_doc_context_read_stops_at_budget_and_preserves_first_cursor():
    # Each page has ~3,500 est_tokens -> 5 pages = 17,500 > MAX_READ_TOKENS (6,000)
    big_lines = "\n".join(f"Line {i:03d}: " + ("revenue metric disclosure text " * 6) for i in range(1, 80))
    doc = PdfDocument.from_pages([big_lines for _ in range(5)])
    ctx = DocContext.from_document(doc)
    res = ctx.read([1, 2, 3, 4, 5], max_chars_per_page=8000)
    assert res["truncated"] is True
    assert res["est_tokens"] <= 6050
    # Must NOT have leaked 1-line stubs of pages 4 or 5 after hitting budget
    assert len(res["pages"]) <= 3
    assert res["next"] is not None
    assert res["next"]["pages"][0] <= 3

    # Duplicate read ledger includes explicit force=true hint
    short_ctx = DocContext.from_document(PdfDocument.from_pages(["Line A\nLine B"]))
    short_ctx.read([1])
    dup = short_ctx.read([1])
    assert dup["pages"][0]["already_returned"] is True
    assert "force=true" in dup["pages"][0]["hint"]


def test_p0_5_audit_compliance_read_all_does_not_truncate_after_page_6():
    pages = [f"Page {i} standard boilerplate." for i in range(1, 10)]
    pages[7] = "Page 8 Certificate: ISO-27001 valid through 2028."
    doc = PdfDocument.from_pages(pages)
    seen_prompt = {}

    class _CapLLM:
        model = "test"

        def chat_json(self, purpose, system, user, max_tokens=360, images_b64=None):
            seen_prompt["user"] = user
            return {
                "material_required": True,
                "offer_kind": "material",
                "covered": True,
                "against": False,
                "form_issue": False,
                "verbatim_quote": "ISO-27001 valid through 2028",
                "cited_pages": [8],
                "reason": "Provided on page 8.",
            }, "{}"

        def snapshot_stats(self):
            return {"calls": 1}

    v = FactAdjudicator(_CapLLM()).audit_compliance_item(
        doc, "c1", "ISO Cert", "Must provide ISO-27001 cert", list(range(1, 10)), mandatory=True, route="read_all"
    )
    assert "[Page 8]" in seen_prompt["user"]
    assert v.status == "pass"
    assert v.quote_verified is True


def test_p0_6_cached_llm_never_caches_empty_parse_and_preserves_thinking_on_transient_err(tmp_path):
    llm = CachedLLM(cache_dir=tmp_path / "cache", base_url="http://localhost:9999/v1", model="m1", api_key="k", offline=False)

    calls = []

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("transient timeout")
            msg = types.SimpleNamespace(content="not json at all truncated...")
            choice = types.SimpleNamespace(message=msg, finish_reason="length")
            usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=50, prompt_tokens_details=None)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    llm._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions()))
    parsed, _ = llm.chat_json("test_purpose", "sys", "usr", max_tokens=50)
    assert parsed == {}
    # Both attempts must keep extra_body={"thinking": {"type": "disabled"}} on transient non-400 errors!
    assert len(calls) == 2
    assert calls[0].get("extra_body") == {"thinking": {"type": "disabled"}}
    assert calls[1].get("extra_body") == {"thinking": {"type": "disabled"}}
    # Empty parse failure MUST NOT be written to disk cache!
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_p1_mm_self_attestation_blocked_and_trailing_period_na_abstains():
    doc = PdfDocument.from_pages(["Company Alpha reported 2024 revenue of 100 million."])

    class _SelfAttestVLM:
        model = "test-vlm"

        def __init__(self, reply):
            self.reply = reply

        def chat_json(self, *args, **kwargs):
            return dict(self.reply), "{}"

        def snapshot_stats(self):
            return {"calls": 1}

    # Hallucinated observation mentioning "ZorgCorp" must NOT attest "ZorgCorp" in the document!
    v1 = FactAdjudicator(_SelfAttestVLM({
        "observations": [{"page": 1, "detail": "ZorgCorp had 500 revenue"}],
        "evidence_status": "stated",
        "entity_time_match": True,
        "derivation": "",
        "formula": "",
        "reasoning": "Seen.",
        "final_answer": "500",
        "cited_pages": [1],
    })).answer_multimodal(doc, "What was ZorgCorp revenue in 2024?", [1], zoom=False)
    assert v1.status == "abstain"
    assert "MM_UNATTESTED_TERMS" in v1.rule_trace

    # Trailing period in "Not answerable." must still abstain!
    v2 = FactAdjudicator(_SelfAttestVLM({
        "observations": [],
        "evidence_status": "stated",
        "entity_time_match": True,
        "derivation": "",
        "formula": "",
        "reasoning": "Cannot find.",
        "final_answer": "Not answerable.",
        "cited_pages": [1],
    })).answer_multimodal(doc, "What was Alpha margin in 2024?", [1], zoom=False)
    assert v2.status == "abstain"
    assert "MM_MODEL_ABSTAINED" in v2.rule_trace


def test_p1_grounding_scope_unions_localized_and_cited_and_ignores_boilerplate():
    doc = PdfDocument.from_pages([
        "in millions — table header only.",
        "in millions — Revenue 1,000.",
        "in millions — Cost 400.",
        "in millions — Notes.",
    ])
    # "in millions" appears on 4 pages (>3) -> ignored as non-discriminative!
    # Quote on page 2 + cited page 3 -> union {2, 3} grounds both 1000 and 400!
    pages, scope = FactAdjudicator._grounding_scope(
        doc,
        cited=[3],
        verified_quotes=["in millions", "Revenue 1,000"],
        route="read_all",
        fallback=[1, 2, 3, 4],
    )
    assert pages == [2, 3]
    assert scope == "quote_and_cited"
