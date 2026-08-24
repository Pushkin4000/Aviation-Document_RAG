from langchain_core.documents import Document

from app.config import Settings
from app.generation.groq_generator import generate_with_groq
from app.models import REFUSAL_MESSAGE, GenerationOutcome, RetrievedChunk


def _chunk(cid="s.pdf:p1:c1"):
    return RetrievedChunk(
        document=Document(
            page_content="A cold front occurs when cold air replaces warm air.",
            metadata={"source": "s.pdf", "page": 1, "chunk_id": cid},
        ),
        score=0.9,
    )


def _settings():
    return Settings(_env_file=None, groq_api_key="test-key", generation_mode="groq")


def _patch_chain(monkeypatch, response):
    import app.generation.groq_generator as mod
    monkeypatch.setattr(mod, "_invoke_chain", lambda *a, **k: response)


def test_returns_unavailable_without_api_key():
    settings = Settings(_env_file=None, groq_api_key=None)
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", settings)
    assert result.outcome is GenerationOutcome.UNAVAILABLE


def test_model_refusal_returns_declined_not_unavailable(monkeypatch):
    """Regression for bug #1.

    The original returned (REFUSAL_MESSAGE, []) here. The caller tested
    `if answer and used_chunks:` — the empty list is falsy — so it fell
    through to extraction and answered anyway, discarding the refusal.
    """
    _patch_chain(monkeypatch, {"answer": REFUSAL_MESSAGE, "cited_chunk_ids": []})
    result = generate_with_groq("What is the capital of France?", [_chunk()], "m", _settings())
    assert result.outcome is GenerationOutcome.DECLINED
    assert result.answer == REFUSAL_MESSAGE


def test_valid_answer_is_returned_with_cited_chunks(monkeypatch):
    _patch_chain(monkeypatch, {"answer": "A cold front occurs when cold air replaces warm air.", "cited_chunk_ids": ["s.pdf:p1:c1"]})
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())
    assert result.outcome is GenerationOutcome.ANSWERED
    assert [c.chunk_id for c in result.chunks] == ["s.pdf:p1:c1"]


def test_hallucinated_chunk_ids_are_rejected(monkeypatch):
    _patch_chain(monkeypatch, {"answer": "Something.", "cited_chunk_ids": ["does-not-exist:p9:c9"]})
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())
    assert result.outcome is GenerationOutcome.UNAVAILABLE


def test_transport_error_returns_unavailable(monkeypatch):
    import app.generation.groq_generator as mod

    def _boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(mod, "_invoke_chain", _boom)
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())
    assert result.outcome is GenerationOutcome.UNAVAILABLE


def test_empty_answer_returns_unavailable(monkeypatch):
    _patch_chain(monkeypatch, {"answer": "   ", "cited_chunk_ids": ["s.pdf:p1:c1"]})
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())
    assert result.outcome is GenerationOutcome.UNAVAILABLE
