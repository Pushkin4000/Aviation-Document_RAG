"""Unit tests for evaluate.py's build_report — pure function of two
DataFrames, no engine/network required."""
import pandas as pd

from evaluate import build_report


def _row(**overrides):
    base = {
        "id": 1,
        "type": "factual",
        "question": "What is QNH?",
        "expected_answer": "QNH is ...",
        "answer": "QNH is ...",
        "citations": "fixture.pdf (Page 1)",
        "retrieved_chunk_ids": "fixture.pdf:p1:c1",
        "route": "simple",
        "confidence": 0.9,
        "decision": "answer",
        "generation_path": "extractive",
        "refused": False,
        "unanswerable": False,
        "retrieval_hit": True,
        "correct": True,
        "faithful": True,
        "hallucination": False,
        "judge_method": "lexical-fallback",
        "judge_reason": "ok",
        "latency_ms": 100.0,
    }
    base.update(overrides)
    return base


def _oos_row(refused=True):
    return {"id": 1, "question": "cake?", "answer": "", "decision": "", "confidence": 0.0, "refused": refused}


def test_imp1_qualifier_keys_off_rows_not_settings_generation_mode():
    """IMP-1: RAG_GENERATION_MODE=groq with no/invalid key means every
    call actually falls back to extraction (generation_path=="extractive"
    per row), even though settings.generation_mode == "groq". The old code
    computed extractive = generation_mode != "groq" and would have printed
    a bare 100% faithfulness here. The qualifier must still appear."""
    df = pd.DataFrame([_row(generation_path="extractive")])
    oos = pd.DataFrame([_oos_row()])
    report = build_report(df, oos, method="lexical-fallback", generation_mode="groq")
    assert "NOT MEANINGFUL" in report
    assert "Why faithfulness reads" in report


def test_imp1_qualifier_absent_when_rows_actually_used_groq():
    df = pd.DataFrame([_row(generation_path="groq")])
    oos = pd.DataFrame([_oos_row()])
    report = build_report(df, oos, method="llm:x", generation_mode="groq")
    assert "NOT MEANINGFUL" not in report


def test_imp2_refusal_precision_denominator_excludes_unrefused_oos():
    # 1 correct in-scope refusal (unanswerable, refused), 1 out-of-scope
    # question refused, 1 out-of-scope question NOT refused. Precision
    # must be computed over refusals actually issued: 2/2 = 100%, not
    # 2/3 (66.7%) which would count the unrefused oos question too.
    df = pd.DataFrame([
        _row(refused=True, unanswerable=True, correct=True, retrieval_hit=False,
             decision="refuse_low_confidence"),
    ])
    oos = pd.DataFrame([_oos_row(refused=True), _oos_row(refused=False)])
    report = build_report(df, oos, method="rule-only", generation_mode="extractive")
    assert "1/1 (100.0%)" in report or "2/2 (100.0%)" in report


def test_imp3_retrieval_recall_reports_answerable_subset_separately():
    df = pd.DataFrame([
        _row(id=1, retrieval_hit=True, unanswerable=False),
        _row(id=2, retrieval_hit=False, unanswerable=True, refused=True, correct=True),
    ])
    oos = pd.DataFrame([_oos_row()])
    report = build_report(df, oos, method="rule-only", generation_mode="extractive")
    # all-questions figure (1 hit / 2 total = 50%) and answerable-only
    # figure (1 hit / 1 answerable = 100%) must both appear, distinctly.
    assert "50.0%" in report
    assert "100.0%" in report
    assert "answerable subset only" in report
