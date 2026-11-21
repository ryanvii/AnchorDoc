"""Unit tests for BM25, RRF fusion, and programmatic CiteOrDie quote verification."""
from anchordoc.runtime import BM25Okapi, rrf, tokenize, verify_quote


def test_bm25_and_rrf_ranking():
    docs = [
        "Consolidated Balance Sheets Total Assets 12000 million",
        "Item 1 Business Overview and Software Products",
        "Consolidated Statements of Cash Flows Capital Expenditures 1577 million",
    ]
    bm = BM25Okapi([tokenize(d) for d in docs])
    scores = bm.get_scores(tokenize("capital expenditures cash flows"))
    assert scores[2] > scores[0] and scores[2] > scores[1]

    fused = rrf([3, 1, 2], [2, 3, 1], k=60)
    assert fused[0] == 3


def test_verify_quote_exact_ellipsis_and_paraphrase_rejection():
    src = (
        "Section 4.1 Confidentiality. Receiving Party shall not disclose any "
        "Confidential Information to third parties without prior written consent."
    )
    ok, score = verify_quote("Receiving Party shall not disclose any Confidential Information", src)
    assert ok is True and score == 1.0

    ok_ell, score_ell = verify_quote(
        "Receiving Party shall not disclose ... without prior written consent",
        src,
    )
    assert ok_ell is True and score_ell == 0.95

    ok_fake, _ = verify_quote("Recipient may freely share secrets with competitors", src)
    assert ok_fake is False


def test_bilingual_tokenizer():
    toks = tokenize("Consolidated 营业执照 2026年 财务报表")
    assert "consolidated" in toks
    assert "营" in toks and "业" in toks
