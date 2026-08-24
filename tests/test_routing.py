from app.config import Settings
from app.routing import route_question, route_question_heuristic

SETTINGS = Settings(_env_file=None)


def test_short_factual_is_simple():
    assert route_question_heuristic("What is QNH?") == "simple"
    assert route_question_heuristic("What does VOR stand for?") == "simple"


def test_questions_clearing_the_threshold_are_complex():
    # "how" (0.28) + keyword (0.25) + " and " (0.08) = 0.61
    assert route_question_heuristic(
        "How should I compare the trade-off between range and endurance?"
    ) == "complex"
    # "why" (0.28) + " and " (0.08) = 0.36
    assert route_question_heuristic(
        "Why does carburettor icing form and what should the pilot do?"
    ) == "complex"


def test_bare_causal_question_falls_just_short_of_complex():
    """Characterisation test for a known limitation of the inherited heuristic.

    A plain "Why ...?" scores only 0.28 against a 0.32 threshold, so it routes
    as `simple`. This is documented rather than fixed: the weights are carried
    over unchanged so routing stays comparable across the refactor. Impact is
    currently nil because RAG_MODEL_ROUTING_ENABLED defaults to 0, making the
    route metadata only. Revisit as a separate, measured change.
    """
    assert route_question_heuristic("Why does carburettor icing form at high humidity?") == "simple"
    assert route_question_heuristic(
        "If cumulative delays reduce your fuel reserve near legal minimums, what should you do?"
    ) == "simple"


def test_route_always_returns_a_valid_label():
    for q in ["", "x", "What is a cold front?", "Compare the trade-off between range and endurance"]:
        assert route_question(q, SETTINGS) in {"simple", "complex"}


def test_llm_router_is_not_called_when_disabled(monkeypatch):
    import app.routing as mod

    def _fail(*a, **k):
        raise AssertionError("LLM router must not be called when router_llm_enabled is False")

    monkeypatch.setattr(mod, "_route_with_llm", _fail)
    assert route_question("Why is this complex and conditional and long enough?", SETTINGS) == "complex"


def test_llm_router_failure_falls_back_to_heuristic(monkeypatch):
    import app.routing as mod

    monkeypatch.setattr(mod, "_route_with_llm", lambda *a, **k: None)
    settings = Settings(_env_file=None, router_mode="groq", router_llm_enabled=True, groq_api_key="k")
    assert route_question("What is QNH?", settings) == "simple"
