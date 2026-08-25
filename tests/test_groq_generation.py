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


class _RateLimitError(Exception):
    """Mimics groq's RateLimitError, which is matched by class name."""

    def __init__(self, retry_after=None):
        super().__init__("rate limit reached")
        if retry_after is not None:
            self.retry_after = retry_after


_RateLimitError.__name__ = "RateLimitError"


def test_short_rate_limit_is_retried_then_succeeds(monkeypatch):
    """A brief per-minute limit should not cost us the model's answer."""
    import app.generation.groq_generator as mod

    calls = {"n": 0}
    slept = []

    def _flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimitError(retry_after=2.0)
        return {"answer": "A cold front occurs when cold air replaces warm air.", "cited_chunk_ids": ["s.pdf:p1:c1"]}

    monkeypatch.setattr(mod, "_invoke_chain", _flaky)
    monkeypatch.setattr(mod.time, "sleep", slept.append)
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())

    assert result.outcome is GenerationOutcome.ANSWERED
    assert calls["n"] == 2
    assert slept == [2.0]


def test_daily_cap_rate_limit_falls_back_without_stalling(monkeypatch):
    """The tokens-per-day 429 asks for minutes. Waiting that long would hang
    the request, so we fall back to extraction immediately."""
    import app.generation.groq_generator as mod

    slept = []
    monkeypatch.setattr(mod, "_invoke_chain", lambda *a, **k: (_ for _ in ()).throw(_RateLimitError(retry_after=478.0)))
    monkeypatch.setattr(mod.time, "sleep", slept.append)
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())

    assert result.outcome is GenerationOutcome.UNAVAILABLE
    assert slept == []


def test_rate_limit_gives_up_after_max_retries(monkeypatch):
    import app.generation.groq_generator as mod

    calls = {"n": 0}

    def _always_limited(*a, **k):
        calls["n"] += 1
        raise _RateLimitError(retry_after=1.0)

    monkeypatch.setattr(mod, "_invoke_chain", _always_limited)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    settings = Settings(_env_file=None, groq_api_key="test-key", generation_mode="groq", groq_max_retries=2)
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", settings)

    assert result.outcome is GenerationOutcome.UNAVAILABLE
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    import app.generation.groq_generator as mod

    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("connection reset")

    monkeypatch.setattr(mod, "_invoke_chain", _boom)
    result = generate_with_groq("What is a cold front?", [_chunk()], "m", _settings())

    assert result.outcome is GenerationOutcome.UNAVAILABLE
    assert calls["n"] == 1
