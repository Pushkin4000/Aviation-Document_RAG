from app.config import Settings


def test_defaults_match_documented_values():
    s = Settings(_env_file=None)
    assert s.top_k == 8
    assert s.min_relevance == 0.35
    assert s.confidence_answer_threshold == 0.48
    assert s.confidence_clarify_threshold == 0.34
    assert s.generation_mode == "groq"
    assert s.embedding_dimension == 384


def test_env_aliases_are_read(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_CONFIDENCE_ANSWER_THRESHOLD", "0.9")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    s = Settings(_env_file=None)
    assert s.top_k == 3
    assert s.confidence_answer_threshold == 0.9
    assert s.groq_api_key == "test-key"


def test_groq_enabled_requires_key_and_mode(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Settings(_env_file=None).groq_enabled is False
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert Settings(_env_file=None).groq_enabled is True
    monkeypatch.setenv("RAG_GENERATION_MODE", "extractive")
    assert Settings(_env_file=None).groq_enabled is False


def test_clarify_threshold_must_not_exceed_answer_threshold(monkeypatch):
    import pytest
    monkeypatch.setenv("RAG_CONFIDENCE_CLARIFY_THRESHOLD", "0.9")
    monkeypatch.setenv("RAG_CONFIDENCE_ANSWER_THRESHOLD", "0.5")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_settings_can_be_overridden_directly():
    s = Settings(_env_file=None, top_k=2, min_relevance=0.9)
    assert s.top_k == 2
    assert s.min_relevance == 0.9
