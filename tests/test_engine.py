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


def test_groq_answer_is_graded_against_abstractive_thresholds(engine, monkeypatch):
    """CRIT-1(a): the grounding gate must use the abstractive pair only when
    the answer actually came from Groq, inferred from which generator
    produced the result (not just from config). This paraphrase fails the
    extractive similarity threshold (0.65) but clears the abstractive one
    (0.45), and clears both overlap thresholds (0.58)."""
    import app.engine as mod
    from langchain_core.documents import Document
    from app.models import GenerationResult, RetrievedChunk

    cold_front_chunk = Document(
        page_content=(
            "A cold front occurs when cold air replaces warm air at the surface, marked by "
            "a sharp change in wind direction and a fall in temperature."
        ),
        metadata={"source": "fixture.pdf", "page": 2, "chunk_id": "fixture.pdf:p2:c1"},
    )
    paraphrase = (
        "Warm air is replaced by cold air, giving a sharp wind direction "
        "change and a temperature fall at the front surface."
    )
    chunks = [RetrievedChunk(document=cold_front_chunk, score=0.9)]

    engine.settings = engine.settings.model_copy(update={"groq_api_key": "k", "generation_mode": "groq"})
    monkeypatch.setattr(
        mod, "generate_with_groq", lambda *a, **k: GenerationResult.answered(paraphrase, chunks)
    )
    r = engine.ask("What is a cold front?")
    assert r["answer"] == paraphrase
    assert r["decision"] == Decision.ANSWER.value


def test_same_paraphrase_fails_grounding_under_extractive_generation(engine, monkeypatch):
    """Mirror of the test above with generation forced through the
    extractive path: the identical paraphrase must be REFUSED, proving the
    strict/loose split is keyed on the real generation path, not a blanket
    loosening."""
    import app.engine as mod
    from langchain_core.documents import Document
    from app.models import GenerationResult, RetrievedChunk

    cold_front_chunk = Document(
        page_content=(
            "A cold front occurs when cold air replaces warm air at the surface, marked by "
            "a sharp change in wind direction and a fall in temperature."
        ),
        metadata={"source": "fixture.pdf", "page": 2, "chunk_id": "fixture.pdf:p2:c1"},
    )
    paraphrase = (
        "Warm air is replaced by cold air, giving a sharp wind direction "
        "change and a temperature fall at the front surface."
    )
    chunks = [RetrievedChunk(document=cold_front_chunk, score=0.9)]

    monkeypatch.setattr(
        mod, "generate_extractive", lambda *a, **k: GenerationResult.answered(paraphrase, chunks)
    )
    r = engine.ask("What is a cold front?")
    assert r["answer"] == REFUSAL_MESSAGE
    assert r["decision"] == Decision.REFUSE_GROUNDING_FAILED.value


def test_index_state_reports_lexical_only(engine):
    state = engine.index_state
    assert state.vector_loaded is False
    assert state.lexical_entries == 4
