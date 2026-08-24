from typing import Any

from langchain_community.embeddings import SentenceTransformerEmbeddings

from app.config import Settings
from app.logging_setup import get_logger

logger = get_logger("embeddings")


class EmbeddingsUnavailable(RuntimeError):
    """The embedding model could not be loaded, or does not match the index.

    Raised rather than swallowed: an unusable embedder means semantic
    retrieval is dead, and the service must say so instead of quietly
    degrading to lexical-only matching.
    """


def build_embeddings(settings: Settings) -> Any:
    """Load the sentence-transformers embedder.

    Raises EmbeddingsUnavailable with a remediation hint on any failure.
    """
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"local_files_only": settings.hf_local_files_only},
        )
    except Exception as exc:
        hint = (
            f"Could not load embedding model '{settings.embedding_model}'. "
            f"HF_LOCAL_FILES_ONLY={int(settings.hf_local_files_only)}. "
        )
        if settings.hf_local_files_only:
            hint += (
                "Offline mode is on but the model is not in the local HuggingFace "
                "cache. Set HF_LOCAL_FILES_ONLY=0 once so it can be downloaded and "
                "cached, then set it back to 1 for offline runs."
            )
        else:
            hint += "Check network access to huggingface.co."
        logger.error("%s underlying error: %s: %s", hint, type(exc).__name__, exc)
        raise EmbeddingsUnavailable(hint) from exc

    logger.info("Loaded embedding model %s", settings.embedding_model)
    return embeddings


def verify_dimension(embeddings: Any, expected_dim: int, index_path: str) -> int:
    """Confirm the embedder's output width matches the persisted index.

    A mismatch means every similarity score would be meaningless, so this
    fails loudly at startup rather than at query time.
    """
    try:
        probe = embeddings.embed_query("dimension probe")
    except Exception as exc:
        raise EmbeddingsUnavailable(
            f"Embedding model loaded but could not embed text: {type(exc).__name__}: {exc}"
        ) from exc

    actual = len(probe)
    if actual != expected_dim:
        raise EmbeddingsUnavailable(
            f"Embedding dimension mismatch: model produces {actual}-d vectors but the "
            f"index at '{index_path}' expects {expected_dim}-d. The index was built with "
            f"a different model; rebuild it with 'python -m app.ingest --rebuild' or "
            f"restore the original embedding model."
        )
    logger.info("Embedding dimension verified: %d", actual)
    return actual
