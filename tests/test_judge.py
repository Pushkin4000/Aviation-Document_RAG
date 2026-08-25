from app.config import Settings
from app.judge import Verdict, judge

NO_KEY = Settings(_env_file=None, groq_api_key=None)


def test_fallback_marks_matching_answer_correct():
    v = judge(
        question="What is QNH?",
        answer="QNH is the altimeter subscale setting for elevation above mean sea level.",
        expected_answer="QNH is the altimeter subscale setting which indicates elevation above mean sea level.",
        key_facts=["altimeter subscale setting", "mean sea level"],
        cited_context="QNH is the altimeter subscale setting ... elevation above mean sea level.",
        settings=NO_KEY,
    )
    assert v.correct is True
    assert v.method == "lexical-fallback"
    assert v.reason


def test_fallback_marks_wrong_answer_incorrect():
    v = judge(
        question="What is QNH?",
        answer="QNH is the temperature lapse rate in the troposphere.",
        expected_answer="QNH is the altimeter subscale setting which indicates elevation above mean sea level.",
        key_facts=["altimeter subscale setting", "mean sea level"],
        cited_context="QNH is the altimeter subscale setting.",
        settings=NO_KEY,
    )
    assert v.correct is False


def test_fallback_detects_unfaithful_answer():
    v = judge(
        question="What is QNH?",
        answer="QNH must be set to 1013 hPa above the transition altitude in all territories.",
        expected_answer="QNH is the altimeter subscale setting for elevation above mean sea level.",
        key_facts=["altimeter subscale setting"],
        cited_context="QNH is the altimeter subscale setting.",
        settings=NO_KEY,
    )
    assert v.faithful is False


def test_empty_context_is_never_faithful():
    v = judge("q", "some answer", "expected", ["fact"], "", NO_KEY)
    assert v.faithful is False


def test_llm_failure_falls_back(monkeypatch):
    import app.judge as mod

    monkeypatch.setattr(mod, "_judge_with_llm", lambda *a, **k: None)
    settings = Settings(_env_file=None, groq_api_key="k")
    v = judge("q", "a", "a", ["a"], "a", settings)
    assert v.method == "lexical-fallback"
    assert isinstance(v, Verdict)
