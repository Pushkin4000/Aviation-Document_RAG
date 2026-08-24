import pytest

from app.config import Settings
from app.embeddings import EmbeddingsUnavailable, build_embeddings, verify_dimension


class _FakeEmbeddings:
    def __init__(self, dim):
        self._dim = dim

    def embed_query(self, text):
        return [0.0] * self._dim


def test_verify_dimension_accepts_match():
    assert verify_dimension(_FakeEmbeddings(384), expected_dim=384, index_path="vectorstore") == 384


def test_verify_dimension_rejects_mismatch():
    with pytest.raises(EmbeddingsUnavailable) as exc:
        verify_dimension(_FakeEmbeddings(768), expected_dim=384, index_path="vectorstore")
    assert "768" in str(exc.value) and "384" in str(exc.value)


def test_build_embeddings_raises_with_hint_when_model_missing(monkeypatch):
    """The exact live failure: model absent from cache, offline mode forced."""
    import app.embeddings as mod

    def _boom(*args, **kwargs):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr(mod, "SentenceTransformerEmbeddings", _boom)
    settings = Settings(_env_file=None, hf_local_files_only=True)
    with pytest.raises(EmbeddingsUnavailable) as exc:
        build_embeddings(settings)
    message = str(exc.value)
    assert "HF_LOCAL_FILES_ONLY" in message
    assert "all-MiniLM-L6-v2" in message


def test_build_embeddings_does_not_return_none_on_failure(monkeypatch):
    """Regression guard: failure must raise, never yield a silently broken object."""
    import app.embeddings as mod

    monkeypatch.setattr(mod, "SentenceTransformerEmbeddings", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(EmbeddingsUnavailable):
        build_embeddings(Settings(_env_file=None))
