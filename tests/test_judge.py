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


def test_cache_key_includes_judge_path_so_a_valid_key_is_not_shadowed_by_stale_cache(
    monkeypatch, tmp_path
):
    """MIN-4: a lexical-fallback verdict cached under a no-key run must not
    be silently replayed once a valid GROQ_API_KEY is supplied for the same
    question/answer/expected triple -- the second call must actually reach
    the LLM path."""
    import app.judge as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "cache")

    question, answer, expected = "What is QNH?", "an answer", "an expected answer"

    v1 = judge(question, answer, expected, ["fact"], "context text", NO_KEY)
    assert v1.method == "lexical-fallback"

    called = {"n": 0}

    def fake_llm(*a, **k):
        called["n"] += 1
        return Verdict(correct=True, faithful=True, reason="llm said so", method="llm:fake-model")

    monkeypatch.setattr(mod, "_judge_with_llm", fake_llm)
    keyed_settings = Settings(_env_file=None, groq_api_key="valid-key")
    v2 = judge(question, answer, expected, ["fact"], "context text", keyed_settings)

    assert called["n"] == 1, "the LLM judge path must actually run, not be shadowed by the stale cache entry"
    assert v2.method == "llm:fake-model"


def test_llm_failure_falls_back(monkeypatch):
    import app.judge as mod

    monkeypatch.setattr(mod, "_judge_with_llm", lambda *a, **k: None)
    settings = Settings(_env_file=None, groq_api_key="k")
    v = judge("q", "a", "a", ["a"], "a", settings)
    assert v.method == "lexical-fallback"
    assert isinstance(v, Verdict)
