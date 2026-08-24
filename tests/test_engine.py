from app.models import REFUSAL_MESSAGE, Decision


def test_empty_question_is_refused(engine):
    r = engine.ask("   ")
    assert r["answer"] == REFUSAL_MESSAGE
    assert r["decision"] == Decision.REFUSE_EMPTY_QUESTION.value


def test_out_of_scope_question_is_refused(engine):
    r = engine.ask("What is the best recipe for chocolate cake?")
    assert r["answer"] == REFUSAL_MESSAGE
    assert r["decision"].startswith("refuse")
    assert r["citations"] == []


def test_in_scope_question_is_answered_with_citations(engine):
    r = engine.ask("What is QNH?")
    assert r["answer"] != REFUSAL_MESSAGE
    assert r["citations"]
    assert "fixture.pdf" in r["citations"][0]


def test_debug_exposes_retrieved_chunks(engine):
    assert "retrieved_chunks" not in engine.ask("What is QNH?", debug=False)
    assert engine.ask("What is QNH?", debug=True)["retrieved_chunks"]


def test_every_response_carries_telemetry(engine):
    r = engine.ask("What is a cold front?")
    assert r["route"] in {"simple", "complex"}
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["decision"]


def test_model_refusal_is_not_overridden_by_extraction(engine, monkeypatch):
    """End-to-end regression for bug #1."""
    import app.engine as mod
    from app.models import GenerationResult

    engine.settings = engine.settings.model_copy(update={"groq_api_key": "k", "generation_mode": "groq"})
    monkeypatch.setattr(mod, "generate_with_groq", lambda *a, **k: GenerationResult.declined())
    r = engine.ask("What is QNH?")
    assert r["answer"] == REFUSAL_MESSAGE
    assert r["decision"] == Decision.REFUSE_MODEL_DECLINED.value


def test_index_state_reports_lexical_only(engine):
    state = engine.index_state
    assert state.vector_loaded is False
    assert state.lexical_entries == 4
