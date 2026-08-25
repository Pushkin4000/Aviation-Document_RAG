from functools import lru_cache
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Paths and corpus
    data_dir: str = Field(default="data", validation_alias="RAG_DATA_DIR")
    vectorstore_dir: str = Field(default="vectorstore", validation_alias="RAG_VECTORSTORE_DIR")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", validation_alias="RAG_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, validation_alias="RAG_EMBEDDING_DIMENSION")
    hf_local_files_only: bool = Field(default=False, validation_alias="HF_LOCAL_FILES_ONLY")

    # Retrieval
    top_k: int = Field(default=8, ge=1, le=50, validation_alias="RAG_TOP_K")
    min_relevance: float = Field(default=0.35, ge=0.0, le=1.0, validation_alias="RAG_MIN_RELEVANCE")
    min_chunk_lexical_overlap: float = Field(
        default=0.12, ge=0.0, le=1.0, validation_alias="RAG_MIN_CHUNK_LEXICAL_OVERLAP"
    )
    min_segment_score: float = Field(default=0.23, ge=0.0, le=1.0, validation_alias="RAG_MIN_SEGMENT_SCORE")
    max_candidate_chunks: int = Field(default=5, ge=1, le=20, validation_alias="RAG_MAX_CANDIDATE_CHUNKS")

    # Grounding
    min_grounded_similarity: float = Field(
        default=0.65, ge=0.0, le=1.0, validation_alias="RAG_MIN_GROUNDED_SIMILARITY"
    )
    min_grounded_token_overlap: float = Field(
        default=0.58, ge=0.0, le=1.0, validation_alias="RAG_MIN_GROUNDED_TOKEN_OVERLAP"
    )
    # Abstractive (Groq) pair. The extractive pair above is calibrated for
    # verbatim spans and must not move. A Groq answer is paraphrased, not
    # copied, so it needs its own, looser pair.
    #
    # Calibrated against 37 real Groq answers (not reference answers) on
    # 2026-08-25, the first run after a working API key arrived. Each answer
    # was scored twice: against the chunks it actually cited (grounded) and
    # against chunks retrieved for an unrelated question (ungrounded). The
    # two distributions separate cleanly on token overlap and barely at all
    # on similarity:
    #
    #   overlap    grounded: min 0.24  median 0.55
    #              ungrounded: median 0.09  MAX 0.33
    #   similarity grounded: min 0.48   ungrounded: max 0.56  (overlapping)
    #
    # So overlap does the discriminating and similarity is only a weak
    # backstop. 0.35 is the highest overlap threshold that admits 0.0% of
    # ungrounded answers, and it admits 81.1% of grounded ones. The previous
    # 0.58 admitted just 37.8% -- it rejected 14 of 50 correct, well-cited
    # answers in the first end-to-end Groq run, wiping out the entire
    # 'reasoning' question category.
    min_grounded_similarity_abstractive: float = Field(
        default=0.45, ge=0.0, le=1.0, validation_alias="RAG_MIN_GROUNDED_SIMILARITY_ABSTRACTIVE"
    )
    min_grounded_token_overlap_abstractive: float = Field(
        default=0.35, ge=0.0, le=1.0, validation_alias="RAG_MIN_GROUNDED_TOKEN_OVERLAP_ABSTRACTIVE"
    )
    answer_max_words: int = Field(default=65, ge=10, le=500, validation_alias="RAG_ANSWER_MAX_WORDS")

    # Confidence gates
    confidence_answer_threshold: float = Field(
        default=0.48, ge=0.0, le=1.0, validation_alias="RAG_CONFIDENCE_ANSWER_THRESHOLD"
    )
    confidence_clarify_threshold: float = Field(
        default=0.34, ge=0.0, le=1.0, validation_alias="RAG_CONFIDENCE_CLARIFY_THRESHOLD"
    )
    low_confidence_support_threshold: float = Field(
        default=0.66, ge=0.0, le=1.0, validation_alias="RAG_LOW_CONFIDENCE_SUPPORT_THRESHOLD"
    )

    # Generation
    generation_mode: str = Field(default="groq", validation_alias="RAG_GENERATION_MODE")
    groq_api_key: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", validation_alias="RAG_GROQ_MODEL")
    groq_timeout_seconds: float = Field(default=30.0, gt=0, validation_alias="RAG_GROQ_TIMEOUT_SECONDS")
    # Groq's free tier enforces a tokens-per-day cap. When it is hit, every
    # call returns 429 and generation silently degrades to extraction --
    # which is safe, but reads as "the model answered badly" rather than
    # "the model never ran". A short bounded retry absorbs the brief
    # per-minute limits without letting a request hang: the daily cap
    # reports a multi-minute retry-after, which exceeds the cap below, so
    # we give up immediately and fall back rather than stalling the caller.
    groq_max_retries: int = Field(default=2, ge=0, le=8, validation_alias="RAG_GROQ_MAX_RETRIES")
    groq_retry_max_wait_seconds: float = Field(
        default=10.0, ge=0.0, validation_alias="RAG_GROQ_RETRY_MAX_WAIT_SECONDS"
    )

    # Routing
    router_mode: str = Field(default="heuristic", validation_alias="RAG_ROUTER_MODE")
    router_llm_enabled: bool = Field(default=False, validation_alias="RAG_ROUTER_LLM_ENABLED")
    routing_models_enabled: bool = Field(default=False, validation_alias="RAG_MODEL_ROUTING_ENABLED")
    simple_model: str = Field(default="openai/gpt-oss-20b", validation_alias="RAG_SIMPLE_MODEL")
    complex_model: str = Field(default="openai/gpt-oss-120b", validation_alias="RAG_COMPLEX_MODEL")

    # Evaluation
    judge_model: str = Field(default="openai/gpt-oss-120b", validation_alias="RAG_JUDGE_MODEL")

    # Ingestion
    chunk_size: int = Field(default=900, ge=100, validation_alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, ge=0, validation_alias="RAG_CHUNK_OVERLAP")
    filter_exercise_chunks: bool = Field(default=True, validation_alias="RAG_FILTER_EXERCISE_CHUNKS")

    # Logging
    log_level: str = Field(default="INFO", validation_alias="RAG_LOG_LEVEL")

    @property
    def groq_enabled(self) -> bool:
        """True when the Groq generation path is both configured and keyed."""
        return bool(self.groq_api_key) and self.generation_mode.strip().lower() == "groq"

    @model_validator(mode="after")
    def _check_threshold_ordering(self) -> "Settings":
        if self.confidence_clarify_threshold > self.confidence_answer_threshold:
            raise ValueError(
                "RAG_CONFIDENCE_CLARIFY_THRESHOLD "
                f"({self.confidence_clarify_threshold}) must not exceed "
                f"RAG_CONFIDENCE_ANSWER_THRESHOLD ({self.confidence_answer_threshold})"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
