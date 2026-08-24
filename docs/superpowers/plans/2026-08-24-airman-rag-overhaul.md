# AIRMAN Aviation RAG Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a submitted RAG prototype into a fast, reliable, honestly-measured, deployable service with a frontend that exposes the retrieval pipeline.

**Architecture:** Split the 1,108-line `app/graph.py` god class into single-responsibility modules behind an injected `Settings` object and a lazily-constructed engine. Groq becomes the primary generation path with extraction as fallback. Evaluation is rebuilt against authored ground truth. A static frontend is served directly by FastAPI.

**Tech Stack:** Python 3.13, FastAPI 0.128, pydantic-settings 2.12, FAISS (faiss-cpu 1.13), sentence-transformers 5.2, langchain-community 0.4 / langchain-core 1.2, langchain-groq 1.1, rapidfuzz 3.14, pandas 3.0, pytest 9.0.

**Spec:** `docs/superpowers/specs/2026-08-24-airman-rag-overhaul-design.md`

## Global Constraints

- The refusal string is exactly `This information is not available in the provided document(s).` — never reworded, never punctuation-adjusted. It lives in one constant, `app/models.py::REFUSAL_MESSAGE`.
- Chunk size 900, chunk overlap 150, embedding model `all-MiniLM-L6-v2`, 384 dimensions. The committed index has 6,590 vectors; do not rebuild it.
- Never commit `.env`. Never print or log the value of `GROQ_API_KEY`.
- No bare `except Exception: pass`. Every caught exception is logged with context and either re-raised or converted to an explicit typed state.
- No new runtime dependency beyond what is already installed, except `pydantic-settings` (present) — the embeddings adapter is written locally rather than adding `langchain-huggingface`.
- Every task ends green: `pytest` passes before the commit step.
- Frontend palette tokens are exactly the hex values in the spec's Colour section. No indigo/violet, no slate-900/950, no stock Tailwind semantics, no `R=G=B` neutral, no `#FFFFFF`/`#000000`.
- Tests must not require a `GROQ_API_KEY` or network access.

## Baseline (verified 2026-08-24, before any change)

Recorded so regressions are detectable:

- `from app.graph import rag_engine` takes **47.9s** (index + model load at import).
- `rag_engine.vectorstore is None` is **True** — FAISS never loads. The service answers from lexical search only.
- Cause: `all-MiniLM-L6-v2` is absent from the HuggingFace cache and `HF_LOCAL_FILES_ONLY=1` forbids fetching it. `SentenceTransformerEmbeddings(...)` raises `OSError`, swallowed by `except Exception: self.vectorstore = None`.
- `vectorstore/index.faiss`: 6,590 vectors, dim 384, `IndexFlatL2` — valid, built from all 7 PDFs.
- `"What is QNH in altimetry?"` returns confidence `1.0` with the answer `"QNH is always rounded down to the nearest integer."` — maximal confidence on a poor answer.

---

### Task 1: Repository hygiene and dependency pinning

Removes the standing risk of publishing `.env`, and makes the build reproducible.

**Files:**
- Create: `.gitignore`, `.env.example`, `requirements-dev.txt`
- Modify: `requirements.txt`
- Delete from index (not disk): all tracked `*.pyc`

**Interfaces:**
- Consumes: nothing
- Produces: a clean tree; `pip install -r requirements.txt` reproduces the verified environment

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Secrets
.env
.env.local

# Python
__pycache__/
*.py[cod]
*.pyc.*
*.egg-info/
.pytest_cache/
.ruff_cache/

# Environments
.venv/
venv/

# Source corpus — 527 MB, distributed separately. See README.
data/

# Local artifacts
.eval_cache/
```

- [ ] **Step 2: Untrack the committed bytecode**

```bash
git rm -r --cached --quiet app/__pycache__ __pycache__
git status --short | head -20
```

Expected: ~44 `D` entries for `.pyc` paths, and the files still present on disk.

- [ ] **Step 3: Restore `.env.example` with no values**

```dotenv
# Required for the Groq generation path. Without it the service falls back
# to extractive answering and reports the downgrade on GET /health.
GROQ_API_KEY=

# Generation: groq | extractive
RAG_GENERATION_MODE=groq
RAG_GROQ_MODEL=llama-3.3-70b-versatile

# Retrieval
RAG_TOP_K=8
RAG_MIN_RELEVANCE=0.35
RAG_MIN_CHUNK_LEXICAL_OVERLAP=0.12
RAG_MIN_SEGMENT_SCORE=0.23

# Grounding
RAG_MIN_GROUNDED_SIMILARITY=0.65
RAG_MIN_GROUNDED_TOKEN_OVERLAP=0.58
RAG_ANSWER_MAX_WORDS=65

# Confidence gates
RAG_CONFIDENCE_ANSWER_THRESHOLD=0.48
RAG_CONFIDENCE_CLARIFY_THRESHOLD=0.34
RAG_LOW_CONFIDENCE_SUPPORT_THRESHOLD=0.66

# Routing
RAG_ROUTER_MODE=heuristic
RAG_ROUTER_LLM_ENABLED=0
RAG_MODEL_ROUTING_ENABLED=0
RAG_SIMPLE_MODEL=llama-3.1-8b-instant
RAG_COMPLEX_MODEL=llama-3.3-70b-versatile

# Evaluation judge
RAG_JUDGE_MODEL=llama-3.3-70b-versatile

# Ingestion
RAG_FILTER_EXERCISE_CHUNKS=1

# Set to 1 ONLY when the embedding model is already cached locally.
# Leaving this at 1 with an empty cache is what silently disabled FAISS.
HF_LOCAL_FILES_ONLY=0
```

- [ ] **Step 3b: Set `HF_LOCAL_FILES_ONLY=0` in the local `.env`**

Edit `.env` (untracked) and change `HF_LOCAL_FILES_ONLY=1` to `HF_LOCAL_FILES_ONLY=0`. This is what allows the embedding model to be fetched on first run. Do not commit `.env`.

- [ ] **Step 4: Pin `requirements.txt` to the verified versions**

```
fastapi==0.128.7
uvicorn==0.40.0
python-dotenv==1.2.1
pydantic==2.12.5
pydantic-settings==2.12.0
langchain-community==0.4.1
langchain-core==1.2.11
langchain-text-splitters==1.1.0
langchain-groq==1.1.2
faiss-cpu==1.13.2
sentence-transformers==5.2.2
pypdf==6.7.0
pandas==3.0.0
rapidfuzz==3.14.3
```

- [ ] **Step 5: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest==9.0.2
httpx==0.28.1
```

- [ ] **Step 6: Verify nothing sensitive is staged**

```bash
git add -A
git diff --cached --name-only | grep -E '^\.env$' && echo "STOP: .env staged" || echo "safe: .env not staged"
```

Expected: `safe: .env not staged`

- [ ] **Step 7: Commit**

```bash
git commit -m "chore: add gitignore, untrack bytecode, pin dependencies"
```

---

### Task 2: Configuration and logging foundation

Replaces twenty import-time globals with one injectable object, and makes failures visible.

**Files:**
- Create: `app/config.py`, `app/logging_setup.py`, `tests/__init__.py`, `tests/test_config.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `app.config.Settings` — pydantic-settings model, all fields below
  - `app.config.get_settings() -> Settings` — cached accessor
  - `app.logging_setup.configure_logging(level: str = "INFO") -> None`
  - `app.logging_setup.get_logger(name: str) -> logging.Logger`

- [ ] **Step 1: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 4: Write `app/config.py`**

Note the field naming: `RAG_MODEL_ROUTING_ENABLED` maps to `routing_models_enabled`, not `model_routing_enabled`, because pydantic reserves the `model_` prefix.

```python
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
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="RAG_GROQ_MODEL")
    groq_timeout_seconds: float = Field(default=30.0, gt=0, validation_alias="RAG_GROQ_TIMEOUT_SECONDS")

    # Routing
    router_mode: str = Field(default="heuristic", validation_alias="RAG_ROUTER_MODE")
    router_llm_enabled: bool = Field(default=False, validation_alias="RAG_ROUTER_LLM_ENABLED")
    routing_models_enabled: bool = Field(default=False, validation_alias="RAG_MODEL_ROUTING_ENABLED")
    simple_model: str = Field(default="llama-3.1-8b-instant", validation_alias="RAG_SIMPLE_MODEL")
    complex_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="RAG_COMPLEX_MODEL")

    # Evaluation
    judge_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="RAG_JUDGE_MODEL")

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS, 5 tests

- [ ] **Step 6: Write `app/logging_setup.py`**

```python
import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a single stderr handler. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("app")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"app.{name}")
```

- [ ] **Step 7: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/config.py app/logging_setup.py tests/ pytest.ini
git commit -m "feat: add Settings object and logging setup"
```

---

### Task 3: Shared domain models

Defines the vocabulary every later module uses, including the typed generation outcome that fixes the refusal-override bug.

**Files:**
- Create: `app/models.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `REFUSAL_MESSAGE: str`
  - `Decision(str, Enum)` with members listed below
  - `GenerationOutcome(str, Enum)`: `ANSWERED`, `DECLINED`, `UNAVAILABLE`
  - `RetrievedChunk` dataclass: `document`, `score`, `lexical_overlap`; properties `source`, `page`, `chunk_id`
  - `GenerationResult` dataclass: `outcome`, `answer: Optional[str]`, `chunks: List[RetrievedChunk]`; classmethods `answered(answer, chunks)`, `declined()`, `unavailable()`
  - `AskResult` dataclass: `answer`, `citations`, `route`, `confidence`, `decision`, `follow_up_question`, `retrieved`; method `to_payload(debug, top_k)`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:

```python
from langchain_core.documents import Document

from app.models import (
    REFUSAL_MESSAGE,
    AskResult,
    Decision,
    GenerationOutcome,
    GenerationResult,
    RetrievedChunk,
)


def _chunk(text="Cold fronts replace warm air.", page=3, source="met.pdf", score=0.8):
    return RetrievedChunk(
        document=Document(
            page_content=text,
            metadata={"source": source, "page": page, "chunk_id": f"{source}:p{page}:c1"},
        ),
        score=score,
    )


def test_refusal_message_is_exact():
    assert REFUSAL_MESSAGE == "This information is not available in the provided document(s)."


def test_chunk_exposes_metadata():
    c = _chunk()
    assert c.source == "met.pdf"
    assert c.page == 3
    assert c.chunk_id == "met.pdf:p3:c1"


def test_chunk_tolerates_missing_metadata():
    c = RetrievedChunk(document=Document(page_content="x", metadata={}), score=0.1)
    assert c.source == "Unknown"
    assert c.page == 0


def test_declined_is_distinguishable_from_unavailable():
    declined = GenerationResult.declined()
    unavailable = GenerationResult.unavailable()
    assert declined.outcome is GenerationOutcome.DECLINED
    assert unavailable.outcome is GenerationOutcome.UNAVAILABLE
    assert declined.outcome is not unavailable.outcome
    # Both carry no chunks; the outcome is the only signal. This is the
    # distinction the old tuple-return collapsed, causing bug #1.
    assert declined.chunks == [] and unavailable.chunks == []


def test_answered_carries_answer_and_chunks():
    c = _chunk()
    r = GenerationResult.answered("A cold front replaces warm air.", [c])
    assert r.outcome is GenerationOutcome.ANSWERED
    assert r.chunks == [c]


def test_payload_omits_retrieved_chunks_unless_debug():
    result = AskResult(
        answer="x.",
        citations=["met.pdf (Page 3)"],
        route="simple",
        confidence=0.7123456,
        decision=Decision.ANSWER,
        retrieved=[_chunk()],
    )
    assert "retrieved_chunks" not in result.to_payload(debug=False, top_k=8)
    payload = result.to_payload(debug=True, top_k=8)
    assert payload["retrieved_chunks"][0]["chunk_id"] == "met.pdf:p3:c1"
    assert payload["confidence"] == 0.7123
    assert payload["decision"] == "answer"


def test_payload_omits_empty_follow_up():
    result = AskResult(
        answer=REFUSAL_MESSAGE,
        citations=[],
        route="simple",
        confidence=0.1,
        decision=Decision.REFUSE_LOW_CONFIDENCE,
        follow_up_question=None,
    )
    assert "follow_up_question" not in result.to_payload(debug=False, top_k=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: Write `app/models.py`**

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from langchain_core.documents import Document

REFUSAL_MESSAGE = "This information is not available in the provided document(s)."


class Decision(str, Enum):
    ANSWER = "answer"
    ANSWER_LOW_CONFIDENCE_GROUNDED = "answer_low_confidence_grounded"
    CLARIFY_LOW_CONFIDENCE = "clarify_low_confidence"
    REFUSE_EMPTY_QUESTION = "refuse_empty_question"
    REFUSE_NO_RELEVANT_CHUNKS = "refuse_no_relevant_chunks"
    REFUSE_LOW_CONFIDENCE = "refuse_low_confidence"
    REFUSE_NO_SUPPORTED_ANSWER = "refuse_no_supported_answer"
    REFUSE_GROUNDING_FAILED = "refuse_grounding_failed"
    REFUSE_MODEL_DECLINED = "refuse_model_declined"


class GenerationOutcome(str, Enum):
    ANSWERED = "answered"
    DECLINED = "declined"        # the model judged the context insufficient
    UNAVAILABLE = "unavailable"  # no key, transport error, or unusable response


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    lexical_overlap: float = 0.0

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "Unknown"))

    @property
    def page(self) -> int:
        try:
            return int(self.document.metadata.get("page", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", f"{self.source}:p{self.page}:c?"))

    @property
    def citation(self) -> str:
        return f"{self.source} (Page {self.page})"

    def to_payload(self, snippet_chars: int = 320) -> Dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 4),
            "lexical_overlap": round(self.lexical_overlap, 4),
            "content_snippet": self.document.page_content[:snippet_chars],
        }


@dataclass
class GenerationResult:
    """Result of one generation attempt.

    The three outcomes must stay distinct. Collapsing DECLINED and
    UNAVAILABLE into a falsy value is what caused the model's refusal to be
    silently overridden by extractive answering.
    """

    outcome: GenerationOutcome
    answer: Optional[str] = None
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @classmethod
    def answered(cls, answer: str, chunks: List[RetrievedChunk]) -> "GenerationResult":
        return cls(outcome=GenerationOutcome.ANSWERED, answer=answer, chunks=list(chunks))

    @classmethod
    def declined(cls) -> "GenerationResult":
        return cls(outcome=GenerationOutcome.DECLINED, answer=REFUSAL_MESSAGE, chunks=[])

    @classmethod
    def unavailable(cls) -> "GenerationResult":
        return cls(outcome=GenerationOutcome.UNAVAILABLE, answer=None, chunks=[])


@dataclass
class AskResult:
    answer: str
    citations: List[str]
    route: str
    confidence: float
    decision: Decision
    follow_up_question: Optional[str] = None
    retrieved: List[RetrievedChunk] = field(default_factory=list)

    def to_payload(self, debug: bool, top_k: int) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "answer": self.answer,
            "citations": list(self.citations),
            "route": self.route,
            "confidence": round(self.confidence, 4),
            "decision": self.decision.value,
        }
        if self.follow_up_question:
            payload["follow_up_question"] = self.follow_up_question
        if debug:
            limit = max(3, min(len(self.retrieved), top_k))
            payload["retrieved_chunks"] = [c.to_payload() for c in self.retrieved[:limit]]
        return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add domain models with typed generation outcome"
```

---

### Task 4: Embeddings loader with a dimension guard

**This is the highest-value task in the plan.** It fixes the live failure where FAISS silently never loads, and makes that class of failure impossible to reintroduce.

**Files:**
- Create: `app/embeddings.py`, `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.logging_setup.get_logger`
- Produces:
  - `EmbeddingsUnavailable(RuntimeError)`
  - `build_embeddings(settings) -> Embeddings` — raises `EmbeddingsUnavailable` with a remediation hint instead of returning a broken object
  - `verify_dimension(embeddings, expected_dim, index_path) -> int` — raises `EmbeddingsUnavailable` on mismatch

- [ ] **Step 1: Write the failing test**

`tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.embeddings'`

- [ ] **Step 3: Write `app/embeddings.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embeddings.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Verify against the real environment**

Run:

```bash
python -c "
from app.config import Settings
from app.embeddings import build_embeddings, verify_dimension
s = Settings(_env_file=None, hf_local_files_only=False)
e = build_embeddings(s)
print('dim:', verify_dimension(e, 384, 'vectorstore'))
"
```

Expected: downloads the model on first run, then prints `dim: 384`. If it prints a `EmbeddingsUnavailable` message instead, network access to huggingface.co is required once — this is the fix for the live outage, so do not proceed until it succeeds.

- [ ] **Step 6: Commit**

```bash
git add app/embeddings.py tests/test_embeddings.py
git commit -m "fix: fail loudly when the embedding model is unavailable

The model was absent from the HF cache while HF_LOCAL_FILES_ONLY=1, so
SentenceTransformerEmbeddings raised OSError, which a bare except swallowed.
FAISS then never loaded and every query silently fell back to lexical
matching. Loading now raises with a remediation hint, and the embedder's
output width is checked against the index at startup."
```

---

### Task 5: Scoring module

Extracts the shared arithmetic so the evaluator and the engine tokenize identically. Today they use different stop-word lists.

**Files:**
- Create: `app/scoring.py`, `tests/test_scoring.py`
- Reference: `app/graph.py:47-77` (STOP_WORDS), `:1063-1085` (tokenize/overlap/normalize), `:1043-1055` (noise_penalty), `:415-441` (confidence), `:975-996` (support)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `STOP_WORDS: set[str]`
  - `tokenize(text) -> List[str]`
  - `token_overlap_ratio(question_tokens, text) -> float`
  - `normalize_for_match(text) -> str`
  - `noise_penalty(text) -> float`
  - `extract_acronyms(question) -> List[str]`
  - `definition_expansion_bonus(question, acronyms, text) -> float`
  - `is_definition_question(question) -> bool`
  - `calculate_confidence(question, chunks) -> float`
  - `support_strength(question_tokens, answer, chunks) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:

```python
from langchain_core.documents import Document

from app.models import RetrievedChunk
from app.scoring import (
    calculate_confidence,
    extract_acronyms,
    is_definition_question,
    noise_penalty,
    normalize_for_match,
    support_strength,
    token_overlap_ratio,
    tokenize,
)


def _chunk(text, score=0.8, overlap=0.5):
    return RetrievedChunk(
        document=Document(page_content=text, metadata={"source": "s.pdf", "page": 1}),
        score=score,
        lexical_overlap=overlap,
    )


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("What is the QNH of a runway?") == ["qnh", "runway"]


def test_tokenize_is_case_and_punctuation_insensitive():
    assert tokenize("COLD-FRONT, cold front!") == ["cold", "front", "cold", "front"]


def test_token_overlap_ratio_bounds():
    assert token_overlap_ratio([], "anything") == 0.0
    assert token_overlap_ratio(["qnh"], "the qnh setting") == 1.0
    assert token_overlap_ratio(["qnh", "runway"], "the qnh setting") == 0.5


def test_noise_penalty_flags_exam_style_text():
    clean = "A cold front occurs when cold air replaces warm air."
    exam = "Questions a. one b. two c. three d. four? which is correct?"
    assert noise_penalty(clean) == 0.0
    assert noise_penalty(exam) > 0.2 - 1e-9
    assert noise_penalty(exam) <= 0.22


def test_extract_acronyms():
    assert extract_acronyms("What does VOR and DME mean?") == ["VOR", "DME"]
    assert extract_acronyms("what is a cold front") == []


def test_is_definition_question():
    assert is_definition_question("What is QNH?")
    assert is_definition_question("How is dew point defined?")
    assert not is_definition_question("Why does icing form on the wing?")


def test_confidence_is_zero_without_chunks():
    assert calculate_confidence("anything", []) == 0.0


def test_confidence_is_bounded_and_ordered():
    strong = [_chunk("qnh is the altimeter setting at mean sea level", 0.95, 0.9)]
    weak = [_chunk("unrelated text about catering menus", 0.05, 0.0)]
    hi = calculate_confidence("What is QNH?", strong)
    lo = calculate_confidence("What is QNH?", weak)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi > lo


def test_support_strength_bounded():
    c = _chunk("a cold front replaces warm air", 0.9, 0.8)
    v = support_strength(["cold", "front"], "A cold front replaces warm air.", [c])
    assert 0.0 <= v <= 1.0
    assert support_strength(["x"], "y", []) == 0.0


def test_normalize_for_match_strips_symbols():
    assert normalize_for_match("  QNH:  1013 hPa!! ") == "qnh: 1013 hpa!!"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.scoring'`

- [ ] **Step 3: Write `app/scoring.py`**

Move the following from `app/graph.py` **verbatim**, converting each method to a module-level function by dropping the `self` parameter and replacing every `self._name(...)` call with `name(...)`:

| Source in `graph.py` | New function |
|---|---|
| `STOP_WORDS` (lines 47–77) | `STOP_WORDS` |
| `_tokenize` (1063–1065) | `tokenize` |
| `_token_overlap_ratio` (1067–1072) | `token_overlap_ratio` |
| `_normalize_for_match` (1074–1077) | `normalize_for_match` |
| `_noise_penalty` (1049–1061) | `noise_penalty` |
| `_extract_acronyms` (1024–1025) | `extract_acronyms` |
| `_definition_expansion_bonus` (1027–1047) | `definition_expansion_bonus` |
| `_is_definition_question` (1079–1085) | `is_definition_question` |
| `_calculate_confidence` (415–441) | `calculate_confidence` |
| `_support_strength` (975–996) | `support_strength` |

Header for the new file:

```python
import re
from typing import List

from app.models import RetrievedChunk
```

`calculate_confidence` and `support_strength` take `chunks: List[RetrievedChunk]`. Their bodies are unchanged — the weights stay exactly as they are, because Task 15 measures this configuration as the baseline.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add app/scoring.py tests/test_scoring.py
git commit -m "refactor: extract scoring functions from graph.py"
```

---

### Task 6: Grounding module

**Files:**
- Create: `app/grounding.py`, `tests/test_grounding.py`
- Reference: `app/graph.py:930-973` (`_is_grounded`)

**Interfaces:**
- Consumes: `app.scoring`, `app.models`, `app.config.Settings`
- Produces: `is_grounded(answer, used_chunks, settings) -> bool`

- [ ] **Step 1: Write the failing test**

`tests/test_grounding.py`:

```python
from langchain_core.documents import Document

from app.config import Settings
from app.grounding import is_grounded
from app.models import REFUSAL_MESSAGE, RetrievedChunk

SETTINGS = Settings(_env_file=None)
CONTEXT = (
    "Cold Fronts. If cold air is replacing warm air, then the front is called a "
    "cold front. The passage of a cold front is marked by a sharp change in wind "
    "direction and a fall in temperature."
)


def _chunks(text=CONTEXT):
    return [RetrievedChunk(document=Document(page_content=text, metadata={}), score=0.9)]


def test_refusal_is_always_grounded():
    assert is_grounded(REFUSAL_MESSAGE, [], SETTINGS) is True


def test_verbatim_span_is_grounded():
    assert is_grounded("If cold air is replacing warm air, then the front is called a cold front.", _chunks(), SETTINGS)


def test_invented_content_is_not_grounded():
    assert not is_grounded("A cold front always produces severe hail and tornado activity.", _chunks(), SETTINGS)


def test_answer_without_context_is_not_grounded():
    assert not is_grounded("A cold front replaces warm air.", [], SETTINGS)


def test_question_shaped_answer_is_rejected():
    assert not is_grounded("What is a cold front?", _chunks(), SETTINGS)


def test_multiple_choice_residue_is_rejected():
    assert not is_grounded("The front is a. warm b. cold c. occluded", _chunks(), SETTINGS)


def test_answer_containing_question_mark_is_rejected():
    assert not is_grounded("Is the front called a cold front?", _chunks(), SETTINGS)


def test_mostly_numeric_answer_is_rejected():
    assert not is_grounded("1013 2992 1015 1020 1025 1030 1035 1040", _chunks(), SETTINGS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grounding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.grounding'`

- [ ] **Step 3: Write `app/grounding.py`**

Move `_is_grounded` from `graph.py:930-973` verbatim, with these mechanical changes:
- signature becomes `def is_grounded(answer: str, used_chunks: List[RetrievedChunk], settings: Settings) -> bool:`
- `self._noise_penalty(...)` → `scoring.noise_penalty(...)`
- `self._normalize_for_match(...)` → `scoring.normalize_for_match(...)`
- `self._tokenize(...)` → `scoring.tokenize(...)`
- `MIN_GROUNDED_TOKEN_OVERLAP` → `settings.min_grounded_token_overlap`
- `MIN_GROUNDED_SIMILARITY` → `settings.min_grounded_similarity`

Header:

```python
import re
from typing import List

from rapidfuzz import fuzz

from app import scoring
from app.config import Settings
from app.logging_setup import get_logger
from app.models import REFUSAL_MESSAGE, RetrievedChunk

logger = get_logger("grounding")
```

Add one line immediately before each `return False` in the early-rejection block so failures are traceable — for example:

```python
    if re.search(r"\b[a-d]\.\s", lower_answer):
        logger.debug("grounding rejected: multiple-choice residue")
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_grounding.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add app/grounding.py tests/test_grounding.py
git commit -m "refactor: extract grounding verification from graph.py"
```

---

### Task 7: Routing module

**Files:**
- Create: `app/routing.py`, `tests/test_routing.py`
- Reference: `app/graph.py:372-413`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.scoring.tokenize`
- Produces:
  - `route_question(question, settings) -> str` — returns `"simple"` or `"complex"`
  - `route_question_heuristic(question) -> str`

- [ ] **Step 1: Write the failing test**

`tests/test_routing.py`:

```python
from app.config import Settings
from app.routing import route_question, route_question_heuristic

SETTINGS = Settings(_env_file=None)


def test_short_factual_is_simple():
    assert route_question_heuristic("What is QNH?") == "simple"
    assert route_question_heuristic("What does VOR stand for?") == "simple"


def test_questions_clearing_the_threshold_are_complex():
    # "how" (0.28) + "compare" (0.25) + "trade-off" (0.25) = 0.78
    assert route_question_heuristic(
        "How should I compare the trade-off between range and endurance?"
    ) == "complex"
    # "why" (0.28) + " and " (0.08) = 0.36
    assert route_question_heuristic(
        "Why does carburettor icing form and what should the pilot do?"
    ) == "complex"


def test_bare_causal_question_falls_just_short_of_complex():
    """Characterisation test for a known limitation of the inherited heuristic.

    A plain "Why ...?" scores only 0.28 against the 0.32 threshold, so it
    routes as `simple`. Documented rather than fixed: the weights are carried
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_routing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.routing'`

- [ ] **Step 3: Write `app/routing.py`**

```python
from typing import Optional

from app.config import Settings
from app.logging_setup import get_logger
from app.scoring import tokenize

logger = get_logger("routing")


def route_question_heuristic(question: str) -> str:
    """Classify complexity without any API call.

    Weights are carried over unchanged from the original implementation so
    routing behaviour stays comparable across the refactor.
    """
    q = question.lower().strip()
    q_tokens = tokenize(question)
    complexity = 0.0

    if q.startswith(("why", "how")):
        complexity += 0.28
    if " if " in f" {q} ":
        complexity += 0.2
    if any(term in q for term in ["scenario", "trade-off", "conditional", "implication", "compare"]):
        complexity += 0.25
    if len(q_tokens) > 11:
        complexity += 0.12
    if " and " in q:
        complexity += 0.08
    if " or " in q:
        complexity += 0.06
    if q.count("?") > 1:
        complexity += 0.06

    return "complex" if complexity >= 0.32 else "simple"


def _route_with_llm(question: str, settings: Settings) -> Optional[str]:
    """Ask Groq to classify. Returns None on any failure, so the caller falls back."""
    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq
    except ImportError as exc:
        logger.warning("LLM router unavailable, langchain-groq not importable: %s", exc)
        return None

    try:
        llm = ChatGroq(
            model=settings.simple_model,
            temperature=0,
            api_key=settings.groq_api_key,
            timeout=settings.groq_timeout_seconds,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 'Classify the question complexity. Return JSON: {{"route": "simple" | "complex"}}.'),
                ("human", "{question}"),
            ]
        )
        result = (prompt | llm | JsonOutputParser()).invoke({"question": question})
        route = str(result.get("route", "")).strip().lower()
        if route in {"simple", "complex"}:
            return route
        logger.warning("LLM router returned unusable label %r", route)
    except Exception as exc:
        logger.warning("LLM router failed (%s: %s), using heuristic", type(exc).__name__, exc)
    return None


def route_question(question: str, settings: Settings) -> str:
    if settings.router_mode.strip().lower() == "groq" and settings.router_llm_enabled and settings.groq_api_key:
        route = _route_with_llm(question, settings)
        if route:
            return route
    return route_question_heuristic(question)
```

Note: the JSON braces in the system prompt are doubled (`{{...}}`) because `ChatPromptTemplate` treats single braces as variable slots. The original at `graph.py:381` uses single braces and would raise on any input — this is a latent bug fixed here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_routing.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add app/routing.py tests/test_routing.py
git commit -m "refactor: extract query router, fix unescaped prompt braces"
```

---

### Task 8: Retrieval module

Also fixes bug #2: the `idx < 2` clause that made the no-evidence refusal path unreachable.

**Files:**
- Create: `app/retrieval.py`, `tests/test_retrieval.py`
- Reference: `app/graph.py:245-370` (retrieve, lexical, select)

**Interfaces:**
- Consumes: `app.config.Settings`, `app.scoring`, `app.models.RetrievedChunk`, `app.embeddings`
- Produces:
  - `LexicalIndex` class: `__init__(entries)`, `from_docstore(docstore)`, `search(question, top_k, settings) -> List[RetrievedChunk]`, `__len__`
  - `Retriever` class: `__init__(vectorstore, lexical_index, settings)`, `retrieve(question) -> List[RetrievedChunk]`, `select_candidates(question, retrieved) -> Tuple[List[RetrievedChunk], List[str]]`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval.py`:

```python
from langchain_core.documents import Document

from app.config import Settings
from app.models import RetrievedChunk
from app.retrieval import LexicalIndex, Retriever

SETTINGS = Settings(_env_file=None)


def _doc(text, page=1, cid="s.pdf:p1:c1"):
    return Document(page_content=text, metadata={"source": "s.pdf", "page": page, "chunk_id": cid})


def _index():
    return LexicalIndex(
        [
            _doc("QNH is the altimeter subscale setting to obtain elevation above mean sea level.", 1, "s.pdf:p1:c1"),
            _doc("A cold front occurs when cold air replaces warm air at the surface.", 2, "s.pdf:p2:c1"),
            _doc("Catering arrangements for long haul cabin service.", 3, "s.pdf:p3:c1"),
        ]
    )


def test_lexical_search_ranks_relevant_first():
    hits = _index().search("What is QNH?", top_k=3, settings=SETTINGS)
    assert hits
    assert "QNH" in hits[0].document.page_content


def test_lexical_search_returns_nothing_for_unrelated_query():
    assert _index().search("chocolate cake recipe", top_k=3, settings=SETTINGS) == []


def test_retriever_uses_lexical_when_vectorstore_absent():
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=SETTINGS)
    assert r.retrieve("What is QNH?")


def test_select_candidates_returns_empty_for_irrelevant_chunks():
    """Regression for bug #2.

    The original admitted any chunk at idx < 2 regardless of score or
    overlap, so candidates were never empty and the no-evidence refusal
    branch was unreachable.
    """
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=SETTINGS)
    junk = [
        RetrievedChunk(document=_doc("Catering arrangements for cabin service."), score=0.01, lexical_overlap=0.0),
        RetrievedChunk(document=_doc("Baggage handling procedures at the ramp."), score=0.01, lexical_overlap=0.0),
    ]
    candidates, tokens = r.select_candidates("What is the boiling point of water?", junk)
    assert candidates == []
    assert tokens


def test_select_candidates_keeps_relevant_chunks():
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=SETTINGS)
    good = [RetrievedChunk(document=_doc("QNH is the altimeter subscale setting."), score=0.9)]
    candidates, _ = r.select_candidates("What is QNH?", good)
    assert len(candidates) == 1


def test_select_candidates_respects_max_candidates():
    settings = Settings(_env_file=None, max_candidate_chunks=2)
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=settings)
    many = [RetrievedChunk(document=_doc("QNH altimeter setting mean sea level."), score=0.9) for _ in range(6)]
    candidates, _ = r.select_candidates("What is QNH?", many)
    assert len(candidates) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval'`

- [ ] **Step 3: Write `app/retrieval.py`**

Port `_lexical_retrieve` (`graph.py:307-342`) into `LexicalIndex.search` and `_retrieve` (`graph.py:245-271`) into `Retriever.retrieve`, replacing `self._name(...)` with `scoring.name(...)` and module globals with `settings.*`. `LexicalIndex.from_docstore` replaces the manual `pickle.load` at `graph.py:273-305`:

```python
@classmethod
def from_docstore(cls, docstore) -> "LexicalIndex":
    """Build from a loaded FAISS docstore instead of unpickling index.pkl by hand."""
    raw = list(getattr(docstore, "_dict", {}).values())
    return cls([d for d in raw if isinstance(d, Document) and d.page_content])
```

`select_candidates` is `_select_candidate_chunks` (`graph.py:344-370`) with the bug fixed — the admission test loses `or idx < 2`:

```python
def select_candidates(self, question, retrieved):
    question_tokens = scoring.tokenize(question)
    if not retrieved:
        return [], question_tokens

    candidates = []
    for item in retrieved:
        overlap = scoring.token_overlap_ratio(question_tokens, item.document.page_content)
        item.lexical_overlap = overlap
        # Admission is on evidence alone. The original also admitted the top
        # two chunks unconditionally (`or idx < 2`), which meant this list was
        # never empty and Decision.REFUSE_NO_RELEVANT_CHUNKS was unreachable.
        if item.score >= self.settings.min_relevance or overlap >= self.settings.min_chunk_lexical_overlap:
            candidates.append(item)

    if not candidates:
        logger.info("No candidate chunks cleared the relevance gate for %r", question[:80])
        return [], question_tokens

    candidates.sort(
        key=lambda c: (0.6 * c.score)
        + (0.35 * c.lexical_overlap)
        - (0.2 * scoring.noise_penalty(c.document.page_content[:420])),
        reverse=True,
    )
    return candidates[: self.settings.max_candidate_chunks], question_tokens
```

In `Retriever.retrieve`, replace the nested bare-except fallback at `graph.py:250-260` with logged handling:

```python
try:
    results = self.vectorstore.similarity_search_with_relevance_scores(question, k=self.settings.top_k)
    vector_results = [
        RetrievedChunk(document=doc, score=max(0.0, min(float(score or 0.0), 1.0)))
        for doc, score in results
    ]
except Exception as exc:
    logger.warning("Vector search failed (%s: %s), using lexical only", type(exc).__name__, exc)
    vector_results = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "refactor: extract retrieval; make the no-evidence refusal reachable

select_candidates admitted the top two chunks regardless of score or
overlap, so the candidate list was never empty and
REFUSE_NO_RELEVANT_CHUNKS could never fire. Admission is now on evidence
alone."
```

---

### Task 9: Extractive generation

The fallback path, moved wholesale so its behaviour is preserved and testable.

**Files:**
- Create: `app/generation/__init__.py`, `app/generation/extractive.py`, `tests/test_extractive.py`
- Reference: `app/graph.py:522-928`

**Interfaces:**
- Consumes: `app.scoring`, `app.config.Settings`, `app.models`
- Produces: `generate_extractive(question, question_tokens, chunks, settings) -> GenerationResult`

- [ ] **Step 1: Write the failing test**

`tests/test_extractive.py`:

```python
from langchain_core.documents import Document

from app.config import Settings
from app.generation.extractive import clean_answer_text, generate_extractive
from app.models import GenerationOutcome, RetrievedChunk

SETTINGS = Settings(_env_file=None)


def _chunk(text):
    return RetrievedChunk(
        document=Document(page_content=text, metadata={"source": "s.pdf", "page": 1, "chunk_id": "s.pdf:p1:c1"}),
        score=0.9,
        lexical_overlap=0.8,
    )


def test_extracts_a_definition():
    chunks = [_chunk("QNH is the altimeter subscale setting to obtain elevation above mean sea level when on the ground.")]
    result = generate_extractive("What is QNH?", ["qnh"], chunks, SETTINGS)
    assert result.outcome is GenerationOutcome.ANSWERED
    assert "qnh" in result.answer.lower()
    assert result.chunks


def test_returns_unavailable_when_no_chunks():
    result = generate_extractive("What is QNH?", ["qnh"], [], SETTINGS)
    assert result.outcome is GenerationOutcome.UNAVAILABLE
    assert result.answer is None


def test_returns_unavailable_when_nothing_matches():
    chunks = [_chunk("Catering arrangements for long haul cabin service on wide body aircraft.")]
    result = generate_extractive("What is the boiling point of water?", ["boiling", "point", "water"], chunks, SETTINGS)
    assert result.outcome is GenerationOutcome.UNAVAILABLE


def test_clean_answer_truncates_to_max_words():
    settings = Settings(_env_file=None, answer_max_words=5)
    out = clean_answer_text("one two three four five six seven eight", settings)
    assert len(out.rstrip(".").split()) == 5


def test_clean_answer_terminates_sentence():
    assert clean_answer_text("a grounded statement", SETTINGS).endswith(".")


def test_clean_answer_cuts_multiple_choice_tail():
    out = clean_answer_text("The front is cold a. warm b. cold c. occluded", SETTINGS)
    assert "b." not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.generation'`

- [ ] **Step 3: Write `app/generation/__init__.py`**

```python
from app.generation.extractive import generate_extractive
from app.generation.groq_generator import generate_with_groq

__all__ = ["generate_extractive", "generate_with_groq"]
```

- [ ] **Step 4: Write `app/generation/extractive.py`**

Move these from `graph.py`, dropping `self`, routing helper calls to `app.scoring`, and replacing globals with `settings.*`:

| Source | New function |
|---|---|
| `_answer_extractive` (522–599) | `generate_extractive` |
| `_acronym_probe` (601–636) | `_acronym_probe` |
| `_definition_probe` (638–712) | `_definition_probe` |
| `_is_definition_supportive` (714–762) | `_is_definition_supportive` |
| `_is_definition_answer_form` (764–801) | `_is_definition_answer_form` |
| `_segment_text` (803–827) | `_segment_text` |
| `_is_valid_segment` (829–838) | `_is_valid_segment` |
| `_segment_score` (840–869) | `_segment_score` |
| `_keyword_window` (871–905) | `_keyword_window` |
| `_clean_answer_text` (907–928) | `clean_answer_text` (public — tested directly) |
| `_refine_answer_clause` (930-…) | `refine_answer_clause` (public — reused by the Groq path) |

`generate_extractive` keeps its logic but returns a `GenerationResult` instead of a tuple:

```python
def generate_extractive(question, question_tokens, chunks, settings) -> GenerationResult:
    if not chunks:
        return GenerationResult.unavailable()
    # ... body unchanged from _answer_extractive ...
    # every `return answer, [chunk]` becomes:
    #     return GenerationResult.answered(answer, [chunk])
    # the final `return None, []` becomes:
    #     return GenerationResult.unavailable()
```

`clean_answer_text` and `_segment_score` take `settings` as a trailing parameter, replacing `ANSWER_MAX_WORDS` with `settings.answer_max_words` and `MIN_SEGMENT_SCORE` with `settings.min_segment_score`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_extractive.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Commit**

```bash
git add app/generation/ tests/test_extractive.py
git commit -m "refactor: extract extractive generation into its own module"
```

---

### Task 10: Groq generation and the refusal-override fix

**Files:**
- Create: `app/generation/groq_generator.py`, `tests/test_groq_generation.py`
- Reference: `app/graph.py:459-520`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.models`, `app.generation.extractive.clean_answer_text`
- Produces: `generate_with_groq(question, chunks, model_name, settings) -> GenerationResult`

- [ ] **Step 1: Write the failing test**

`tests/test_groq_generation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_groq_generation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `app/generation/groq_generator.py`**

```python
from typing import Any, Dict, List

from app.config import Settings
from app.generation.extractive import clean_answer_text, refine_answer_clause
from app.logging_setup import get_logger
from app.models import REFUSAL_MESSAGE, GenerationResult, RetrievedChunk
from app.scoring import tokenize

logger = get_logger("generation.groq")

SYSTEM_PROMPT = (
    "You answer strictly from the provided context about aviation documents.\n"
    "Use only facts present in the context. Do not add outside knowledge.\n"
    f"If the context does not support an answer, set answer to exactly: {REFUSAL_MESSAGE}\n"
    "Return JSON with keys: answer (string), cited_chunk_ids (list of the Chunk ID "
    "values you actually used). Cite only chunk IDs that appear in the context."
)


def _build_context(chunks: List[RetrievedChunk], limit: int = 5) -> str:
    return "\n\n".join(
        f"Chunk ID: {c.chunk_id}\nSource: {c.source} (Page {c.page})\nContent: {c.document.page_content}"
        for c in chunks[:limit]
    )


def _invoke_chain(question: str, context: str, model_name: str, settings: Settings) -> Dict[str, Any]:
    """Isolated so tests can substitute a response without touching the network."""
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=settings.groq_api_key,
        timeout=settings.groq_timeout_seconds,
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "Question: {question}\n\nContext:\n{context}")]
    )
    return (prompt | llm | JsonOutputParser()).invoke({"question": question, "context": context})


def generate_with_groq(
    question: str,
    chunks: List[RetrievedChunk],
    model_name: str,
    settings: Settings,
) -> GenerationResult:
    if not settings.groq_api_key:
        logger.debug("Groq generation skipped: no API key configured")
        return GenerationResult.unavailable()
    if not chunks:
        return GenerationResult.unavailable()

    try:
        response = _invoke_chain(question, _build_context(chunks), model_name, settings)
    except Exception as exc:
        logger.warning("Groq generation failed (%s: %s)", type(exc).__name__, exc)
        return GenerationResult.unavailable()

    if not isinstance(response, dict):
        logger.warning("Groq returned a non-object response: %r", type(response).__name__)
        return GenerationResult.unavailable()

    answer = str(response.get("answer", "")).strip()
    if not answer:
        logger.warning("Groq returned an empty answer")
        return GenerationResult.unavailable()

    # The model judged the context insufficient. This is a real decision and
    # must be propagated, not treated as a failed call.
    if answer == REFUSAL_MESSAGE:
        logger.info("Groq declined to answer: context judged insufficient")
        return GenerationResult.declined()

    available = {c.chunk_id: c for c in chunks}
    cited = [str(cid) for cid in response.get("cited_chunk_ids", []) or []]
    used = [available[cid] for cid in cited if cid in available]
    if not used:
        logger.warning("Groq cited no valid chunk IDs (returned %r), discarding answer", cited)
        return GenerationResult.unavailable()

    refined = refine_answer_clause(question, tokenize(question), answer, settings)
    return GenerationResult.answered(clean_answer_text(refined, settings), used)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_groq_generation.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add app/generation/groq_generator.py tests/test_groq_generation.py
git commit -m "fix: stop discarding the model's refusal

generate_with_groq returned (REFUSAL_MESSAGE, []) when the model declined.
The caller's truthiness check treated the empty chunk list as failure and
fell through to extractive answering, so a correct refusal became an
answer. Outcomes are now typed: ANSWERED, DECLINED, UNAVAILABLE."
```

---

### Task 11: Engine orchestration with lazy initialisation

**Files:**
- Create: `app/engine.py`, `tests/conftest.py`, `tests/test_engine.py`
- Rewrite: `app/graph.py` (becomes a re-export shim)

**Interfaces:**
- Consumes: every module from Tasks 2–10
- Produces:
  - `IndexState` dataclass: `vector_loaded: bool`, `lexical_entries: int`, `degraded_reason: Optional[str]`
  - `AviationRAGEngine` class: `__init__(settings=None)`, `load()`, `refresh_index()`, `ask(question, debug=False) -> Dict`, `index_state -> IndexState`, `generation_path -> str`
  - `get_engine() -> AviationRAGEngine` — cached, constructs on first call
  - `reset_engine() -> None` — for tests

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.engine import AviationRAGEngine
from app.retrieval import LexicalIndex

CORPUS = [
    ("QNH is the altimeter subscale setting which causes the altimeter to indicate "
     "elevation above mean sea level when the aircraft is on the ground.", 1),
    ("A cold front occurs when cold air replaces warm air at the surface, marked by "
     "a sharp change in wind direction and a fall in temperature.", 2),
    ("VOR stands for VHF Omni-directional Range, a navigation aid used to define "
     "airways and for en-route navigation.", 3),
    ("Catering arrangements for long haul cabin service on wide body aircraft.", 4),
]


@pytest.fixture
def settings():
    return Settings(_env_file=None, groq_api_key=None, generation_mode="extractive")


@pytest.fixture
def engine(settings):
    """An engine backed by an in-memory lexical index. No model, no network."""
    docs = [
        Document(page_content=text, metadata={"source": "fixture.pdf", "page": page,
                                              "chunk_id": f"fixture.pdf:p{page}:c1"})
        for text, page in CORPUS
    ]
    eng = AviationRAGEngine(settings=settings)
    eng.attach_index(vectorstore=None, lexical_index=LexicalIndex(docs))
    return eng
```

- [ ] **Step 2: Write the failing test**

`tests/test_engine.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.engine'`

- [ ] **Step 4: Write `app/engine.py`**

```python
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS

from app import scoring
from app.config import Settings, get_settings
from app.embeddings import EmbeddingsUnavailable, build_embeddings, verify_dimension
from app.generation.extractive import generate_extractive
from app.generation.groq_generator import generate_with_groq
from app.grounding import is_grounded
from app.logging_setup import get_logger
from app.models import (
    REFUSAL_MESSAGE,
    AskResult,
    Decision,
    GenerationOutcome,
    GenerationResult,
    RetrievedChunk,
)
from app.retrieval import LexicalIndex, Retriever
from app.routing import route_question

logger = get_logger("engine")


@dataclass
class IndexState:
    vector_loaded: bool
    lexical_entries: int
    degraded_reason: Optional[str] = None


class AviationRAGEngine:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.vectorstore: Optional[FAISS] = None
        self.lexical_index = LexicalIndex([])
        self.degraded_reason: Optional[str] = None
        self._retriever = Retriever(None, self.lexical_index, self.settings)

    # -- index lifecycle -------------------------------------------------

    def attach_index(self, vectorstore, lexical_index: LexicalIndex) -> None:
        self.vectorstore = vectorstore
        self.lexical_index = lexical_index
        self._retriever = Retriever(vectorstore, lexical_index, self.settings)

    def load(self) -> None:
        """Load FAISS and the lexical index. Records a degraded reason on failure."""
        directory = self.settings.vectorstore_dir
        if not os.path.isdir(directory):
            self.degraded_reason = f"vectorstore directory '{directory}' does not exist"
            logger.error(self.degraded_reason)
            self.attach_index(None, LexicalIndex([]))
            return

        try:
            embeddings = build_embeddings(self.settings)
            verify_dimension(embeddings, self.settings.embedding_dimension, directory)
            store = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
        except EmbeddingsUnavailable as exc:
            self.degraded_reason = str(exc)
            logger.error("Semantic retrieval unavailable: %s", exc)
            self.attach_index(None, LexicalIndex([]))
            return
        except Exception as exc:
            self.degraded_reason = f"FAISS load failed: {type(exc).__name__}: {exc}"
            logger.error(self.degraded_reason)
            self.attach_index(None, LexicalIndex([]))
            return

        lexical = LexicalIndex.from_docstore(store.docstore)
        self.degraded_reason = None
        self.attach_index(store, lexical)
        logger.info("Index loaded: %d vectors, %d lexical entries", store.index.ntotal, len(lexical))

    def refresh_index(self) -> None:
        self.load()

    @property
    def index_state(self) -> IndexState:
        return IndexState(
            vector_loaded=self.vectorstore is not None,
            lexical_entries=len(self.lexical_index),
            degraded_reason=self.degraded_reason,
        )

    @property
    def generation_path(self) -> str:
        return "groq" if self.settings.groq_enabled else "extractive"

    # -- query -----------------------------------------------------------

    def ask(self, question: str, debug: bool = False) -> Dict[str, object]:
        clean = question.strip()
        if not clean:
            return self._respond(
                REFUSAL_MESSAGE, [], [], debug, "simple", 0.0,
                Decision.REFUSE_EMPTY_QUESTION,
                "Please ask a specific question from the provided aviation documents.",
            )

        retrieved = self._retriever.retrieve(clean)
        candidates, question_tokens = self._retriever.select_candidates(clean, retrieved)
        route = route_question(clean, self.settings)
        confidence = scoring.calculate_confidence(clean, candidates)

        if not candidates:
            return self._respond(
                REFUSAL_MESSAGE, [], retrieved, debug, route, confidence,
                Decision.REFUSE_NO_RELEVANT_CHUNKS,
                self._follow_up(question_tokens, candidates),
            )

        if confidence < self.settings.confidence_clarify_threshold:
            return self._respond(
                REFUSAL_MESSAGE, [], retrieved, debug, route, confidence,
                Decision.REFUSE_LOW_CONFIDENCE,
                self._follow_up(question_tokens, candidates),
            )

        if confidence < self.settings.confidence_answer_threshold:
            salvage = generate_extractive(clean, question_tokens, candidates, self.settings)
            if salvage.outcome is GenerationOutcome.ANSWERED and is_grounded(
                salvage.answer, salvage.chunks, self.settings
            ):
                support = scoring.support_strength(question_tokens, salvage.answer, salvage.chunks)
                if support >= self.settings.low_confidence_support_threshold:
                    return self._respond(
                        salvage.answer, self._citations(salvage.chunks), retrieved, debug,
                        route, confidence, Decision.ANSWER_LOW_CONFIDENCE_GROUNDED,
                    )
            return self._respond(
                REFUSAL_MESSAGE, [], retrieved, debug, route, confidence,
                Decision.CLARIFY_LOW_CONFIDENCE,
                self._follow_up(question_tokens, candidates),
            )

        result = self._generate(clean, question_tokens, candidates, route)

        if result.outcome is GenerationOutcome.DECLINED:
            # The model judged the evidence insufficient. Honour that.
            return self._respond(
                REFUSAL_MESSAGE, [], retrieved, debug, route, confidence,
                Decision.REFUSE_MODEL_DECLINED,
                self._follow_up(question_tokens, candidates),
            )

        if result.outcome is not GenerationOutcome.ANSWERED or not result.chunks:
            return self._respond(
                REFUSAL_MESSAGE, [], retrieved, debug, route, confidence,
                Decision.REFUSE_NO_SUPPORTED_ANSWER,
                self._follow_up(question_tokens, candidates),
            )

        if not is_grounded(result.answer, result.chunks, self.settings):
            logger.info("Grounding check rejected the answer for %r", clean[:80])
            return self._respond(
                REFUSAL_MESSAGE, [], retrieved, debug, route, confidence,
                Decision.REFUSE_GROUNDING_FAILED,
                self._follow_up(question_tokens, candidates),
            )

        return self._respond(
            result.answer, self._citations(result.chunks), retrieved, debug,
            route, confidence, Decision.ANSWER,
        )

    def _generate(self, question, question_tokens, chunks, route) -> GenerationResult:
        if self.settings.groq_enabled:
            model = self.settings.groq_model
            if self.settings.routing_models_enabled:
                model = self.settings.simple_model if route == "simple" else self.settings.complex_model
            result = generate_with_groq(question, chunks, model, self.settings)
            # DECLINED and ANSWERED are both real outcomes and are returned as-is.
            # Only UNAVAILABLE — no key, transport error, unusable response —
            # falls back to extraction.
            if result.outcome is not GenerationOutcome.UNAVAILABLE:
                return result
            logger.info("Groq unavailable, falling back to extractive generation")
        return generate_extractive(question, question_tokens, chunks, self.settings)

    # -- helpers ---------------------------------------------------------

    def _citations(self, chunks: List[RetrievedChunk]) -> List[str]:
        seen, out = set(), []
        for chunk in chunks:
            if chunk.citation not in seen:
                seen.add(chunk.citation)
                out.append(chunk.citation)
        return out

    def _follow_up(self, question_tokens: List[str], chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "Please include the specific topic, procedure, or instrument from the provided documents."
        merged = " ".join(c.document.page_content for c in chunks[:2]).lower()
        missing = [t for t in question_tokens if t not in merged and len(t) >= 4][:2]
        if missing:
            return (
                "Please clarify the exact context for "
                + ", ".join(missing)
                + " (for example: phase of flight, rule, or instrument)."
            )
        return (
            "Please clarify the exact procedure or condition you want "
            "(for example: phase of flight, minima type, or regulation context)."
        )

    def _respond(self, answer, citations, retrieved, debug, route, confidence, decision, follow_up=None):
        return AskResult(
            answer=answer,
            citations=citations,
            route=route,
            confidence=confidence,
            decision=decision,
            follow_up_question=follow_up,
            retrieved=retrieved,
        ).to_payload(debug=debug, top_k=self.settings.top_k)


_ENGINE: Optional[AviationRAGEngine] = None


def get_engine() -> AviationRAGEngine:
    """Construct the engine on first use, not at import."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AviationRAGEngine()
        _ENGINE.load()
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    _ENGINE = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Replace `app/graph.py` with a shim**

```python
"""Backwards-compatible re-exports.

The implementation moved into focused modules. This shim keeps the import
paths used by the submitted README and evaluate.py working.
"""

from app.embeddings import build_embeddings  # noqa: F401
from app.engine import AviationRAGEngine, get_engine, reset_engine  # noqa: F401
from app.models import REFUSAL_MESSAGE, Decision, RetrievedChunk  # noqa: F401
from app.retrieval import LexicalIndex, Retriever  # noqa: F401

__all__ = [
    "REFUSAL_MESSAGE",
    "AviationRAGEngine",
    "Decision",
    "LexicalIndex",
    "RetrievedChunk",
    "Retriever",
    "build_embeddings",
    "get_engine",
    "reset_engine",
]
```

Note: the module-level `rag_engine` singleton is deliberately **not** re-exported. It was the import-time side effect costing 47.9s. Task 15 updates `evaluate.py` to call `get_engine()`.

- [ ] **Step 7: Confirm the import cost is gone**

Run:

```bash
python -c "import time; t=time.time(); import app.graph; print('import: %.2fs' % (time.time()-t))"
```

Expected: under 5s, versus the 47.9s baseline.

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add app/engine.py app/graph.py tests/conftest.py tests/test_engine.py
git commit -m "refactor: orchestrate via engine module with lazy initialisation"
```

---

### Task 12: Server wiring, health reporting, and static mount

**Files:**
- Rewrite: `app/server.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `app.engine.get_engine`, `app.ingest.ingest_pipeline`
- Produces: `app` (FastAPI); routes `GET /health`, `POST /ask`, `POST /ingest`, `GET /` (frontend)

- [ ] **Step 1: Write the failing test**

`tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.config import Settings
from app.engine import AviationRAGEngine
from app.models import REFUSAL_MESSAGE
from app.retrieval import LexicalIndex


@pytest.fixture
def client(monkeypatch):
    import app.server as server

    docs = [
        Document(
            page_content="QNH is the altimeter subscale setting which causes the altimeter "
                         "to indicate elevation above mean sea level on the ground.",
            metadata={"source": "fixture.pdf", "page": 1, "chunk_id": "fixture.pdf:p1:c1"},
        )
    ]
    eng = AviationRAGEngine(settings=Settings(_env_file=None, groq_api_key=None, generation_mode="extractive"))
    eng.attach_index(vectorstore=None, lexical_index=LexicalIndex(docs))
    monkeypatch.setattr(server, "get_engine", lambda: eng)
    with TestClient(server.app) as c:
        yield c


def test_health_reports_index_and_generation_path(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["refusal_message"] == REFUSAL_MESSAGE
    assert body["index"]["lexical_entries"] == 1
    assert body["index"]["vector_loaded"] is False
    assert body["generation_path"] == "extractive"


def test_health_is_degraded_without_vectors(client):
    assert client.get("/health").json()["status"] == "degraded"


def test_ask_returns_answer_and_citations(client):
    body = client.post("/ask", json={"question": "What is QNH?"}).json()
    assert body["answer"]
    assert "route" in body and "confidence" in body and "decision" in body


def test_ask_refuses_out_of_scope(client):
    body = client.post("/ask", json={"question": "What is the best chocolate cake recipe?"}).json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["citations"] == []


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_rejects_overlong_question(client):
    assert client.post("/ask", json={"question": "x" * 5000}).status_code == 422


def test_debug_flag_returns_chunks(client):
    body = client.post("/ask", json={"question": "What is QNH?", "debug": True}).json()
    assert body["retrieved_chunks"]


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `/health` lacks the `index` key, `/` returns 404

- [ ] **Step 3: Create a placeholder `web/index.html`**

So the mount resolves before Task 16 builds the real page:

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AIRMAN RAG</title></head>
<body><p>Frontend pending — see Task 16.</p></body></html>
```

- [ ] **Step 4: Write `app/server.py`**

```python
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.engine import get_engine
from app.ingest import IngestSummary, ingest_pipeline
from app.logging_setup import configure_logging, get_logger
from app.models import REFUSAL_MESSAGE

logger = get_logger("server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = get_engine()  # loads the index once, at startup rather than at import
    state = engine.index_state
    if state.degraded_reason:
        logger.warning("Started DEGRADED: %s", state.degraded_reason)
    else:
        logger.info("Started: %d lexical entries, generation via %s",
                    state.lexical_entries, engine.generation_path)
    yield


app = FastAPI(title="AIRMAN Aviation Document RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    debug: bool = False


class RetrievedChunkPayload(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float
    lexical_overlap: Optional[float] = None
    content_snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    route: Optional[str] = None
    confidence: Optional[float] = None
    decision: Optional[str] = None
    follow_up_question: Optional[str] = None
    retrieved_chunks: Optional[List[RetrievedChunkPayload]] = None


@app.get("/health")
def health_check() -> dict:
    engine = get_engine()
    state = engine.index_state
    healthy = state.vector_loaded and state.degraded_reason is None
    return {
        "status": "ok" if healthy else "degraded",
        "index": asdict(state),
        "generation_path": engine.generation_path,
        "refusal_message": REFUSAL_MESSAGE,
    }


@app.post("/ingest")
def run_ingest(rebuild: bool = Query(default=False)) -> dict:
    try:
        summary: IngestSummary = ingest_pipeline(rebuild=rebuild)
    except ValueError as exc:
        logger.warning("Ingest rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    get_engine().refresh_index()
    return {
        "status": "completed",
        "mode": "rebuild" if rebuild else "incremental",
        "summary": asdict(summary),
    }


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = get_engine().ask(request.question, debug=request.debug)
    except Exception as exc:
        logger.exception("Query failed for %r", request.question[:80])
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return AskResponse(**result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS, 8 tests

- [ ] **Step 6: Commit**

```bash
git add app/server.py web/index.html tests/test_api.py
git commit -m "feat: lifespan startup, degraded health reporting, static mount"
```

---

### Task 13: Modernise ingestion

**Files:**
- Modify: `app/ingest.py`
- Create: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `app.config.Settings`, `app.embeddings.build_embeddings`
- Produces: unchanged public surface — `ingest_pipeline(data_dir, vectorstore_dir, rebuild) -> IngestSummary`

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:

```python
from langchain_core.documents import Document

from app.ingest import chunk_fingerprint, clean_text, is_noise_chunk, split_documents


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert "altimeter" in clean_text("alti-\nmeter setting")


def test_clean_text_drops_bare_page_numbers():
    assert clean_text("Chapter 3\n147\nCold Fronts") == "Chapter 3 Cold Fronts"


def test_noise_chunk_detects_exam_material():
    assert is_noise_chunk("Questions a. one b. two c. three d. four")
    assert not is_noise_chunk("A cold front occurs when cold air replaces warm air.")


def test_fingerprint_is_stable_across_whitespace():
    a = Document(page_content="Cold  front\n replaces warm air", metadata={"source": "s.pdf", "page": 1})
    b = Document(page_content="cold front replaces warm air", metadata={"source": "s.pdf", "page": 1})
    assert chunk_fingerprint(a) == chunk_fingerprint(b)


def test_fingerprint_differs_across_pages():
    a = Document(page_content="same text", metadata={"source": "s.pdf", "page": 1})
    b = Document(page_content="same text", metadata={"source": "s.pdf", "page": 2})
    assert chunk_fingerprint(a) != chunk_fingerprint(b)


def test_split_assigns_traceable_chunk_ids():
    docs = [Document(page_content="Cold fronts. " * 200, metadata={"source": "s.pdf", "page": 4})]
    chunks, dropped = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert all(c.metadata["chunk_id"].startswith("s.pdf:p4:c") for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_fingerprint'` (currently `_chunk_fingerprint`)

- [ ] **Step 3: Modify `app/ingest.py`**

1. Rename to public: `_clean_text`→`clean_text`, `_is_noise_chunk`→`is_noise_chunk`, `_chunk_fingerprint`→`chunk_fingerprint`. Update all call sites.
2. Replace the direct `SentenceTransformerEmbeddings(...)` construction in `create_or_update_vectorstore` with `build_embeddings(get_settings())`, so ingestion gets the same loud failure as query time.
3. Delete `_bootstrap_registry_from_index` (lines 152–177). Replace its call site with a load through FAISS:

```python
def _bootstrap_registry_from_index(output_path: Path, embeddings) -> Dict[str, Dict[str, str]]:
    """Rebuild the registry from an existing index without unpickling by hand."""
    if not (output_path / "index.faiss").exists():
        return {}
    try:
        store = FAISS.load_local(str(output_path), embeddings, allow_dangerous_deserialization=True)
    except Exception as exc:
        logger.warning("Could not bootstrap registry from index: %s: %s", type(exc).__name__, exc)
        return {}
    registry = {}
    for doc in getattr(store.docstore, "_dict", {}).values():
        if isinstance(doc, Document):
            registry[chunk_fingerprint(doc)] = {
                "chunk_id": str(doc.metadata.get("chunk_id", "")),
                "source": str(doc.metadata.get("source", "")),
                "page": str(doc.metadata.get("page", "")),
            }
    return registry
```

4. Replace `except Exception: return {}` in `_load_registry` with a logged warning.
5. Add `logger = get_logger("ingest")` and an INFO line per file loaded and at pipeline completion.
6. Remove the now-unused `import pickle`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Verify incremental ingest is still a no-op on the existing corpus**

Run: `python -m app.ingest`

Expected: `chunks_added: 0`, `chunks_skipped_existing: 4959` — matching `vectorstore/manifest.json`. If `chunks_added` is non-zero, the fingerprint changed; stop and investigate before committing.

- [ ] **Step 6: Commit**

```bash
git add app/ingest.py tests/test_ingest.py
git commit -m "refactor: modernise ingestion, drop manual pickle access"
```

---

### Task 14: Author ground truth

**Files:**
- Create: `scripts/author_ground_truth.py`, `out_of_scope_set.json`
- Modify: `evaluation_set.json`

**Interfaces:**
- Consumes: `app.engine.get_engine`
- Produces: `evaluation_set.json` where every entry has `question`, `type`, `expected_answer`, `expected_sources`, `key_facts`

- [ ] **Step 1: Write the retrieval helper**

`scripts/author_ground_truth.py`:

```python
"""Print top retrieved chunks per question, for authoring ground truth.

This does not write ground truth. It surfaces what the corpus actually says
so a human (or the agent) can write `expected_answer` from real source text
rather than from memory.

Usage: python scripts/author_ground_truth.py > ground_truth_worksheet.txt
"""

import json
from pathlib import Path

from app.engine import get_engine


def main() -> None:
    questions = json.loads(Path("evaluation_set.json").read_text(encoding="utf-8"))
    engine = get_engine()
    for i, item in enumerate(questions, start=1):
        question = item["question"]
        chunks = engine._retriever.retrieve(question)[:3]
        print(f"\n{'=' * 78}\n[{i}] ({item['type']}) {question}\n{'=' * 78}")
        for c in chunks:
            print(f"\n--- {c.source} p{c.page} (score {c.score:.3f}) [{c.chunk_id}]")
            print(c.document.page_content[:900])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the worksheet**

Run: `python scripts/author_ground_truth.py > ground_truth_worksheet.txt`

Expected: a file with 50 blocks, each showing three real chunks. Confirm the chunks look like prose from the aviation PDFs. If they are exam-question fragments for most entries, stop — retrieval is still broken and Task 4 did not take effect.

- [ ] **Step 3: Author `evaluation_set.json`**

For each of the 50 entries, read its worksheet block and extend the object. Do not invent facts absent from the chunks. Where the corpus genuinely does not answer a question, set `"expected_answer": null` and `"unanswerable": true` — those become correct-refusal cases.

Shape, using entry 1 as the worked example:

```json
{
  "question": "What is QNH in altimetry?",
  "type": "factual",
  "expected_answer": "QNH is the altimeter subscale setting which causes the altimeter to indicate elevation above mean sea level when the aircraft is on the ground.",
  "expected_sources": [{"source": "Meteorology full book.pdf", "page": 135}],
  "key_facts": ["altimeter subscale setting", "mean sea level"],
  "unanswerable": false
}
```

Rules:
- `expected_answer` — one or two sentences, paraphrasing the retrieved source text.
- `expected_sources` — every page that genuinely contains the answer, not only the top hit. Add pages seen in the worksheet even at rank 2 or 3.
- `key_facts` — two to four short strings a correct answer must contain. These drive the no-API-key fallback scoring, so keep them lexically robust (prefer `"mean sea level"` over `"MSL"`).

- [ ] **Step 4: Write `out_of_scope_set.json`**

Fifteen questions the system must refuse. These test the product's defining behaviour, which currently has zero coverage.

```json
[
  {"question": "What is the best recipe for chocolate cake?", "type": "out_of_scope"},
  {"question": "Who won the 2018 FIFA World Cup?", "type": "out_of_scope"},
  {"question": "What is the capital city of Australia?", "type": "out_of_scope"},
  {"question": "How do I reset my email password?", "type": "out_of_scope"},
  {"question": "What is the current price of Bitcoin?", "type": "out_of_scope"},
  {"question": "Write me a Python function to sort a list.", "type": "out_of_scope"},
  {"question": "What are the symptoms of influenza?", "type": "out_of_scope"},
  {"question": "Who is the chief executive of this airline?", "type": "out_of_scope"},
  {"question": "What is the landing fee at Heathrow this year?", "type": "out_of_scope"},
  {"question": "What was the cause of the 2009 Air France 447 accident?", "type": "out_of_scope"},
  {"question": "How much does a Boeing 787 cost to purchase?", "type": "out_of_scope"},
  {"question": "What is the maintenance schedule for my specific aircraft registration?", "type": "out_of_scope"},
  {"question": "What is today's weather at my departure airport?", "type": "out_of_scope"},
  {"question": "Which airline has the best on-time performance?", "type": "out_of_scope"},
  {"question": "What is the square root of 144?", "type": "out_of_scope"}
]
```

Note the deliberate near-misses: entries 10 through 14 are aviation-flavoured but are not in a PPL/CPL/ATPL theory corpus. A system that refuses only obvious non-aviation questions would pass a weaker set and fail this one.

- [ ] **Step 5: Validate the authored data**

Run:

```bash
python -c "
import json
qs = json.load(open('evaluation_set.json'))
assert len(qs) == 50, len(qs)
missing = [q['question'] for q in qs if 'expected_answer' not in q or 'expected_sources' not in q or 'key_facts' not in q]
assert not missing, missing
answerable = [q for q in qs if not q.get('unanswerable')]
assert all(q['expected_answer'] and q['expected_sources'] and q['key_facts'] for q in answerable)
print('50 questions validated;', len(qs) - len(answerable), 'marked unanswerable')
oos = json.load(open('out_of_scope_set.json'))
assert len(oos) == 15
print('15 out-of-scope questions validated')
"
```

Expected: both lines print with no assertion error.

- [ ] **Step 6: Commit**

```bash
git add evaluation_set.json out_of_scope_set.json scripts/author_ground_truth.py
git commit -m "feat: add ground truth and out-of-scope evaluation sets"
```

---

### Task 15: Rebuild the evaluator

**Files:**
- Rewrite: `evaluate.py`
- Create: `app/judge.py`, `tests/test_judge.py`

**Interfaces:**
- Consumes: `app.engine.get_engine`, `app.config.Settings`, `app.scoring`
- Produces:
  - `app.judge.Verdict` dataclass: `correct: bool`, `faithful: bool`, `reason: str`, `method: str`
  - `app.judge.judge(question, answer, expected_answer, key_facts, cited_context, settings) -> Verdict`
  - `evaluate.py` writing `evaluation_detailed.csv` and `report.md`

- [ ] **Step 1: Write the failing test**

`tests/test_judge.py`:

```python
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


def test_llm_failure_falls_back(monkeypatch):
    import app.judge as mod

    monkeypatch.setattr(mod, "_judge_with_llm", lambda *a, **k: None)
    settings = Settings(_env_file=None, groq_api_key="k")
    v = judge("q", "a", "a", ["a"], "a", settings)
    assert v.method == "lexical-fallback"
    assert isinstance(v, Verdict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.judge'`

- [ ] **Step 3: Write `app/judge.py`**

```python
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rapidfuzz import fuzz

from app.config import Settings
from app.logging_setup import get_logger
from app.scoring import tokenize

logger = get_logger("judge")
CACHE_DIR = Path(".eval_cache")

JUDGE_SYSTEM = (
    "You grade a retrieval-augmented answer about aviation theory.\n"
    "Given the question, the reference answer, and the context the system actually "
    "cited, return JSON with keys:\n"
    '  correct (boolean): does the answer convey the same facts as the reference?\n'
    '  faithful (boolean): is every claim in the answer supported by the cited context?\n'
    '  reason (string, one sentence).\n'
    "An answer may be faithful but incorrect (it quotes the context but misses the "
    "question), or correct but unfaithful (right facts, absent from the cited context). "
    "Judge the two independently."
)


@dataclass
class Verdict:
    correct: bool
    faithful: bool
    reason: str
    method: str


def _cache_key(question: str, answer: str, expected: str) -> str:
    return hashlib.sha256(f"{question}|{answer}|{expected}".encode("utf-8")).hexdigest()[:32]


def _cache_get(key: str) -> Optional[Verdict]:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return Verdict(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.warning("Ignoring unreadable cache entry %s: %s", key, exc)
        return None


def _cache_put(key: str, verdict: Verdict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(verdict.__dict__), encoding="utf-8")


def _judge_with_llm(
    question: str, answer: str, expected_answer: str, cited_context: str, settings: Settings
) -> Optional[Verdict]:
    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.judge_model,
            temperature=0,
            api_key=settings.groq_api_key,
            timeout=settings.groq_timeout_seconds,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", JUDGE_SYSTEM),
                (
                    "human",
                    "Question: {question}\n\nReference answer: {expected}\n\n"
                    "System answer: {answer}\n\nCited context:\n{context}",
                ),
            ]
        )
        result = (prompt | llm | JsonOutputParser()).invoke(
            {"question": question, "expected": expected_answer, "answer": answer, "context": cited_context}
        )
        return Verdict(
            correct=bool(result.get("correct", False)),
            faithful=bool(result.get("faithful", False)),
            reason=str(result.get("reason", "")).strip() or "no reason given",
            method=f"llm:{settings.judge_model}",
        )
    except Exception as exc:
        logger.warning("Judge LLM failed (%s: %s), using lexical fallback", type(exc).__name__, exc)
        return None


def _judge_lexically(
    answer: str, expected_answer: str, key_facts: List[str], cited_context: str
) -> Verdict:
    """Deterministic fallback when no API key is available.

    Weaker than the LLM judge and labelled as such in the report, so
    fallback numbers are never presented as judge-quality numbers.
    """
    lower_answer = answer.lower()
    facts_present = [f for f in key_facts if f.lower() in lower_answer]
    fact_recall = len(facts_present) / max(1, len(key_facts))
    similarity = fuzz.token_set_ratio(answer.lower(), expected_answer.lower()) / 100.0
    correct = fact_recall >= 0.5 or similarity >= 0.72

    if not cited_context.strip():
        return Verdict(False, False, "No cited context to verify against.", "lexical-fallback")

    context_tokens = set(tokenize(cited_context))
    answer_tokens = tokenize(answer)
    supported = sum(1 for t in answer_tokens if t in context_tokens) / max(1, len(answer_tokens))
    faithful = supported >= 0.6

    return Verdict(
        correct=correct,
        faithful=faithful,
        reason=(
            f"key-fact recall {fact_recall:.2f}, similarity to reference {similarity:.2f}, "
            f"token support from cited context {supported:.2f}"
        ),
        method="lexical-fallback",
    )


def judge(
    question: str,
    answer: str,
    expected_answer: str,
    key_facts: List[str],
    cited_context: str,
    settings: Settings,
) -> Verdict:
    key = _cache_key(question, answer, expected_answer or "")
    cached = _cache_get(key)
    if cached:
        return cached

    verdict = None
    if settings.groq_api_key:
        verdict = _judge_with_llm(question, answer, expected_answer or "", cited_context, settings)
    if verdict is None:
        verdict = _judge_lexically(answer, expected_answer or "", key_facts, cited_context)

    _cache_put(key, verdict)
    return verdict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_judge.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Rewrite `evaluate.py`**

Replace the whole file. The important structural changes: it calls `get_engine()` rather than importing a singleton, it uses the `retrieved_chunks` already on the debug response rather than re-running retrieval, and it scores faithfulness against **cited** chunks only.

```python
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.config import get_settings
from app.engine import get_engine
from app.judge import judge
from app.logging_setup import configure_logging, get_logger
from app.models import REFUSAL_MESSAGE

logger = get_logger("evaluate")

EVAL_FILE = "evaluation_set.json"
OUT_OF_SCOPE_FILE = "out_of_scope_set.json"
DETAIL_FILE = "evaluation_detailed.csv"
REPORT_FILE = "report.md"


def _cited_context(response: Dict, engine) -> str:
    """Text of the chunks the answer actually cited, not everything retrieved."""
    citations = set(response.get("citations", []))
    if not citations:
        return ""
    parts = []
    for chunk in response.get("retrieved_chunks", []) or []:
        if f"{chunk['source']} (Page {chunk['page']})" in citations:
            parts.append(chunk["content_snippet"])
    return " ".join(parts)


def _retrieval_hit(response: Dict, expected_sources: List[Dict]) -> bool:
    if not expected_sources:
        return False
    got = {(c["source"], int(c["page"])) for c in response.get("retrieved_chunks", []) or []}
    want = {(str(s["source"]), int(s["page"])) for s in expected_sources}
    return bool(got & want)


def evaluate_in_scope(engine, settings) -> List[Dict]:
    questions = json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))
    rows = []
    for idx, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        started = time.perf_counter()
        response = engine.ask(question, debug=True)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = str(response.get("answer", REFUSAL_MESSAGE))
        refused = answer == REFUSAL_MESSAGE
        unanswerable = bool(item.get("unanswerable", False))
        cited = _cited_context(response, engine)

        if refused:
            correct = unanswerable  # refusing an unanswerable question is correct
            faithful = True
            reason = "Refused." + (" Correct: no answer exists in the corpus." if unanswerable
                                   else " Incorrect: the corpus does contain this.")
            method = "rule"
        else:
            verdict = judge(
                question=question,
                answer=answer,
                expected_answer=str(item.get("expected_answer") or ""),
                key_facts=list(item.get("key_facts") or []),
                cited_context=cited,
                settings=settings,
            )
            correct = verdict.correct and not unanswerable
            faithful = verdict.faithful
            reason = verdict.reason
            method = verdict.method

        rows.append({
            "id": idx,
            "type": item.get("type", "unknown"),
            "question": question,
            "expected_answer": item.get("expected_answer") or "",
            "answer": answer,
            "citations": "; ".join(response.get("citations", [])),
            "retrieved_chunk_ids": "; ".join(
                c["chunk_id"] for c in (response.get("retrieved_chunks") or [])[:3]
            ),
            "route": response.get("route", ""),
            "confidence": response.get("confidence", 0.0),
            "decision": response.get("decision", ""),
            "refused": refused,
            "unanswerable": unanswerable,
            "retrieval_hit": _retrieval_hit(response, item.get("expected_sources") or []),
            "correct": correct,
            "faithful": faithful,
            "hallucination": (not refused) and (not faithful),
            "judge_method": method,
            "judge_reason": reason,
            "latency_ms": round(latency_ms, 1),
        })
        logger.info("[%02d/%d] %s -> %s", idx, len(questions), question[:60], response.get("decision"))
    return rows


def evaluate_out_of_scope(engine) -> List[Dict]:
    path = Path(OUT_OF_SCOPE_FILE)
    if not path.exists():
        return []
    rows = []
    for idx, item in enumerate(json.loads(path.read_text(encoding="utf-8")), start=1):
        question = str(item["question"]).strip()
        response = engine.ask(question, debug=True)
        refused = str(response.get("answer", "")) == REFUSAL_MESSAGE
        rows.append({
            "id": idx,
            "question": question,
            "answer": response.get("answer", ""),
            "decision": response.get("decision", ""),
            "confidence": response.get("confidence", 0.0),
            "refused": refused,
        })
    return rows


def build_report(df: pd.DataFrame, oos: pd.DataFrame, method: str) -> str:
    total = len(df)
    answered = df[~df["refused"]]
    n_answered = len(answered)
    retrieval_rate = df["retrieval_hit"].mean() * 100 if total else 0.0
    correctness = df["correct"].mean() * 100 if total else 0.0
    faithful_rate = answered["faithful"].mean() * 100 if n_answered else 0.0
    halluc_rate = answered["hallucination"].mean() * 100 if n_answered else 0.0

    oos_refused = int(oos["refused"].sum()) if len(oos) else 0
    oos_total = len(oos)
    refusal_recall = (oos_refused / oos_total * 100) if oos_total else 0.0

    refusals = df[df["refused"]]
    correct_refusals = int(refusals["unanswerable"].sum()) if len(refusals) else 0
    all_refusals = len(refusals) + oos_total
    all_correct_refusals = correct_refusals + oos_refused
    refusal_precision = (all_correct_refusals / all_refusals * 100) if all_refusals else 0.0

    lat = df["latency_ms"]
    p50 = statistics.median(lat) if total else 0.0
    p95 = sorted(lat)[int(len(lat) * 0.95)] if total > 1 else (lat.iloc[0] if total else 0.0)

    type_lines = []
    for q_type, group in df.groupby("type"):
        ans = group[~group["refused"]]
        type_lines.append(
            f"- **{q_type}** (n={len(group)}): retrieval {group['retrieval_hit'].mean()*100:.1f}%, "
            f"correct {group['correct'].mean()*100:.1f}%, "
            f"faithful {(ans['faithful'].mean()*100 if len(ans) else 0.0):.1f}%"
        )

    ranked = df.assign(
        score=df["correct"].astype(int) * 2 + df["faithful"].astype(int) + df["retrieval_hit"].astype(int)
    )
    best = ranked.sort_values(["score", "confidence"], ascending=[False, False]).head(5)
    worst = ranked.sort_values(["score", "confidence"], ascending=[True, True]).head(5)

    def block(rows):
        return "\n".join(
            f"- Q: {r['question']}\n"
            f"  Answer: {r['answer']}\n"
            f"  Expected: {r['expected_answer'] or '(none — unanswerable)'}\n"
            f"  Citations: {r['citations'] or 'None'}\n"
            f"  Verdict: correct={r['correct']}, faithful={r['faithful']}, decision={r['decision']}\n"
            f"  Why: {r['judge_reason']}"
            for _, r in rows.iterrows()
        )

    return f"""# Evaluation Report

Grading method: **{method}**

## Methodology

Every question carries a reference answer and source pages authored from the
indexed corpus. Three things are measured separately:

- **Retrieval recall** — did the retriever surface a page that actually
  contains the answer?
- **Correctness** — does the answer convey the reference facts?
- **Faithfulness** — is every claim supported by the chunks the answer
  **cited**? Not by everything retrieved.

That last distinction matters. An earlier version of this report compared the
answer against the chunks it had been copied from, which cannot fail and
reported 100% faithfulness and 0% hallucination. **Those figures were an
artifact of the measurement and are not comparable to the numbers below.**

## Dataset
- In-scope questions: {total}
- Answered: {n_answered}
- Refused: {total - n_answered}
- Out-of-scope questions (must be refused): {oos_total}

## Metrics
- Retrieval recall@k: {retrieval_rate:.1f}%
- Answer correctness: {correctness:.1f}%
- Faithfulness (of answered): {faithful_rate:.1f}%
- Hallucination rate (of answered): {halluc_rate:.1f}%
- Refusal recall (out-of-scope correctly refused): {oos_refused}/{oos_total} ({refusal_recall:.1f}%)
- Refusal precision (refusals that were correct): {refusal_precision:.1f}%
- Latency p50 / p95: {p50:.0f} ms / {p95:.0f} ms

## Metrics by Question Type
{chr(10).join(type_lines) if type_lines else "- none"}

## 5 Best Answers
{block(best)}

## 5 Worst Answers
{block(worst)}
"""


def run_evaluation() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = get_engine()

    state = engine.index_state
    if state.degraded_reason:
        raise SystemExit(
            f"Refusing to evaluate a degraded index: {state.degraded_reason}\n"
            "Results would measure the fallback path, not the system."
        )

    rows = evaluate_in_scope(engine, settings)
    oos_rows = evaluate_out_of_scope(engine)

    df = pd.DataFrame(rows)
    oos = pd.DataFrame(oos_rows)
    df.to_csv(DETAIL_FILE, index=False)

    methods = set(df["judge_method"]) - {"rule"}
    method = ", ".join(sorted(methods)) if methods else "rule-only"
    Path(REPORT_FILE).write_text(build_report(df, oos, method), encoding="utf-8")
    print(f"Saved {DETAIL_FILE} and {REPORT_FILE}")


if __name__ == "__main__":
    run_evaluation()
```

Note the guard in `run_evaluation`: it refuses to run against a degraded index. Evaluating the lexical-only fallback and reporting it as system performance is exactly how the original numbers became misleading.

- [ ] **Step 6: Run the evaluation**

Run: `python evaluate.py`

Expected: 50 log lines, then `Saved evaluation_detailed.csv and report.md`. If it exits with "Refusing to evaluate a degraded index", Task 4's fix is not in effect — resolve that first.

- [ ] **Step 7: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add evaluate.py app/judge.py tests/test_judge.py evaluation_detailed.csv report.md
git commit -m "feat: rebuild evaluation against ground truth

Faithfulness is now judged against the chunks the answer cited rather than
the chunks it was extracted from, which is what made the previous metric
unable to fail. Adds correctness, refusal precision/recall, and latency."
```

---

### Task 16: Frontend

**Files:**
- Create: `web/styles.css`, `web/app.js`
- Rewrite: `web/index.html`

**Interfaces:**
- Consumes: `POST /ask`, `GET /health`
- Produces: the query console described in the spec

- [ ] **Step 1: Write `web/styles.css`**

Tokens are copied exactly from the spec. Do not substitute values.

```css
:root {
  /* Surfaces — warm near-black, ochre-traced. Derived in OKLCH. */
  --ground: #0F0B08;
  --panel: #1B1612;
  --raised: #25211B;
  --hairline: #37322C;
  --rule: #4C4741;

  /* Text — warm off-whites, never #FFFFFF */
  --text: #F0ECE7;
  --text-2: #BBB6B0;
  --muted: #8A857F;
  --faint: #625D57;

  /* Accent — sodium vapour amber. One job: live state + primary action. */
  --accent-100: #51321E;
  --accent-200: #854E20;
  --accent-300: #B56C15;
  --accent-400: #E29019;
  --accent-500: #FAB550;
  --accent-600: #FFD795;

  /* Semantics from the same family, not from a framework */
  --refusal: #B6604E;
  --grounded: #99A668;
  --caution: #D7A03D;

  --sans: Archivo, "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;

  --fast: 120ms;
  --medium: 200ms;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  background: var(--ground);
  color: var(--text);
  font-family: var(--sans);
  font-weight: 400;
}

/* Radius is zero everywhere. Instrument bezels and approach plates are square. */
input, button, section, div, article { border-radius: 0; }

.masthead {
  border-bottom: 1px solid var(--hairline);
  padding: 20px 32px;
  display: flex;
  align-items: baseline;
  gap: 16px;
}

.masthead h1 {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0;
}

.masthead .sub {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--muted);
}

#health {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--muted);
}
#health[data-state="ok"] { color: var(--grounded); }
#health[data-state="degraded"] { color: var(--caution); }

.query {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--hairline);
}

#question {
  flex: 1;
  background: var(--panel);
  border: none;
  border-right: 1px solid var(--hairline);
  color: var(--text);
  font-family: var(--sans);
  font-size: 17px;
  padding: 22px 32px;
  outline: none;
}
#question::placeholder { color: var(--faint); }
#question:focus { background: var(--raised); }

#submit {
  background: var(--accent-400);
  color: var(--ground);
  border: none;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  padding: 0 36px;
  cursor: pointer;
  transition: background var(--fast) ease-out;
}
#submit:hover { background: var(--accent-500); }
#submit:disabled { background: var(--accent-100); color: var(--muted); cursor: not-allowed; }

/* Asymmetric 2/3 - 1/3. Sparse answer, dense telemetry. */
.split { display: grid; grid-template-columns: 2fr 1fr; min-height: 60vh; }

.answer-col { padding: 44px 32px; border-right: 1px solid var(--hairline); }

.stages {
  display: flex;
  gap: 20px;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--faint);
  margin-bottom: 36px;
}
.stage[data-active="true"] { color: var(--accent-400); }
.stage[data-done="true"] { color: var(--muted); }

#answer {
  font-size: 24px;
  line-height: 1.4;
  letter-spacing: -0.01em;
  max-width: 68ch;
  margin: 0 0 32px;
}
#answer[data-refusal="true"] {
  color: var(--refusal);
  font-size: 19px;
  border-left: 2px solid var(--refusal);
  padding-left: 20px;
}

.citations { font-family: var(--mono); font-size: 12px; color: var(--text-2); }
.citations h2 {
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--faint);
  font-weight: 400;
  margin: 0 0 10px;
}
.citations li { list-style: none; padding: 5px 0; border-bottom: 1px solid var(--panel); }
.citations ul { margin: 0; padding: 0; }

/* Telemetry rail — dense, mono, small */
.rail { padding: 44px 24px; font-family: var(--mono); font-size: 11px; }
.rail h2 {
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--faint);
  font-weight: 400;
  margin: 0 0 14px;
}

.metric { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--panel); }
.metric .k { color: var(--muted); letter-spacing: 0.06em; }
.metric .v { color: var(--text); }

/* Confidence against its threshold bands */
.gauge { margin: 22px 0; }
.gauge-track { height: 3px; background: var(--panel); position: relative; }
.gauge-fill { height: 3px; background: var(--accent-400); transition: width var(--medium) ease-out; }
.gauge-tick { position: absolute; top: -4px; width: 1px; height: 11px; background: var(--rule); }
.gauge-labels { display: flex; justify-content: space-between; color: var(--faint); font-size: 9px; margin-top: 6px; letter-spacing: 0.08em; }

.chunk { padding: 11px 0; border-bottom: 1px solid var(--panel); }
.chunk-id { color: var(--muted); font-size: 10px; word-break: break-all; }
.chunk-score { color: var(--text-2); }
.chunk[data-cited="true"] { border-left: 2px solid var(--accent-400); padding-left: 10px; }
.chunk-snippet { color: var(--faint); font-size: 10px; line-height: 1.5; margin-top: 5px; }

/* Nothing animates except the stage indicator. */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}

@media (max-width: 900px) {
  .split { grid-template-columns: 1fr; }
  .answer-col { border-right: none; border-bottom: 1px solid var(--hairline); }
}
```

- [ ] **Step 2: Write `web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIRMAN — Document RAG</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;700&family=IBM+Plex+Mono:wght@400;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/static/styles.css">
</head>
<body>
<header class="masthead">
  <h1>AIRMAN</h1>
  <span class="sub">DOCUMENT-GROUNDED RETRIEVAL</span>
  <span id="health" data-state="unknown">CHECKING…</span>
</header>

<form class="query" id="form">
  <input id="question" name="question" autocomplete="off" required maxlength="2000"
         placeholder="Ask about altimetry, navigation, mass and balance, air regulation…">
  <button id="submit" type="submit">ASK</button>
</form>

<main class="split">
  <section class="answer-col">
    <div class="stages" id="stages">
      <span class="stage" data-stage="retrieve">RETRIEVE</span>
      <span class="stage" data-stage="route">ROUTE</span>
      <span class="stage" data-stage="generate">GENERATE</span>
      <span class="stage" data-stage="verify">VERIFY</span>
    </div>
    <p id="answer">Ask a question to query the indexed aviation corpus.</p>
    <div class="citations" id="citations"></div>
  </section>

  <aside class="rail">
    <h2>Telemetry</h2>
    <div class="metric"><span class="k">ROUTE</span><span class="v" id="m-route">—</span></div>
    <div class="metric"><span class="k">DECISION</span><span class="v" id="m-decision">—</span></div>
    <div class="metric"><span class="k">LATENCY</span><span class="v" id="m-latency">—</span></div>

    <div class="gauge">
      <h2>Confidence</h2>
      <div class="gauge-track">
        <div class="gauge-fill" id="gauge-fill" style="width:0%"></div>
        <div class="gauge-tick" id="tick-clarify"></div>
        <div class="gauge-tick" id="tick-answer"></div>
      </div>
      <div class="gauge-labels">
        <span>0.00</span><span id="gauge-value">—</span><span>1.00</span>
      </div>
    </div>

    <h2>Retrieved</h2>
    <div id="chunks"></div>
  </aside>
</main>

<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write `web/app.js`**

```js
'use strict';

// Threshold defaults mirror app/config.py. /health could expose these later.
const CLARIFY_THRESHOLD = 0.34;
const ANSWER_THRESHOLD = 0.48;
const REFUSAL = 'This information is not available in the provided document(s).';

const $ = (id) => document.getElementById(id);

function placeTicks() {
  $('tick-clarify').style.left = (CLARIFY_THRESHOLD * 100) + '%';
  $('tick-answer').style.left = (ANSWER_THRESHOLD * 100) + '%';
}

async function loadHealth() {
  const el = $('health');
  try {
    const r = await fetch('/health');
    const body = await r.json();
    el.dataset.state = body.status;
    const vectors = body.index.vector_loaded ? 'VECTOR+LEXICAL' : 'LEXICAL ONLY';
    el.textContent = `${body.status.toUpperCase()} · ${vectors} · ${body.generation_path.toUpperCase()}`;
    el.title = body.index.degraded_reason || '';
  } catch (err) {
    el.dataset.state = 'degraded';
    el.textContent = 'UNREACHABLE';
  }
}

// Motion exists only to report request state.
let stageTimers = [];
function runStages() {
  clearStages();
  const stages = ['retrieve', 'route', 'generate', 'verify'];
  stages.forEach((name, i) => {
    stageTimers.push(setTimeout(() => {
      stages.slice(0, i).forEach((prev) => {
        document.querySelector(`[data-stage="${prev}"]`).dataset.done = 'true';
        document.querySelector(`[data-stage="${prev}"]`).dataset.active = 'false';
      });
      const el = document.querySelector(`[data-stage="${name}"]`);
      if (el) el.dataset.active = 'true';
    }, i * 200));
  });
}

function clearStages() {
  stageTimers.forEach(clearTimeout);
  stageTimers = [];
  document.querySelectorAll('.stage').forEach((el) => {
    el.dataset.active = 'false';
    el.dataset.done = 'false';
  });
}

function renderChunks(chunks, citations) {
  const host = $('chunks');
  host.textContent = '';
  if (!chunks || !chunks.length) {
    host.innerHTML = '<div class="chunk-snippet">No chunks retrieved.</div>';
    return;
  }
  const cited = new Set(citations || []);
  chunks.forEach((c) => {
    const div = document.createElement('div');
    div.className = 'chunk';
    div.dataset.cited = cited.has(`${c.source} (Page ${c.page})`) ? 'true' : 'false';

    const id = document.createElement('div');
    id.className = 'chunk-id';
    id.textContent = c.chunk_id;

    const score = document.createElement('div');
    score.className = 'chunk-score';
    score.textContent = `score ${c.score.toFixed(3)} · overlap ${(c.lexical_overlap ?? 0).toFixed(3)}`;

    const snippet = document.createElement('div');
    snippet.className = 'chunk-snippet';
    snippet.textContent = (c.content_snippet || '').slice(0, 190);

    div.append(id, score, snippet);
    host.appendChild(div);
  });
}

function renderCitations(citations) {
  const host = $('citations');
  host.textContent = '';
  if (!citations || !citations.length) return;
  const h = document.createElement('h2');
  h.textContent = 'Sources';
  const ul = document.createElement('ul');
  citations.forEach((c) => {
    const li = document.createElement('li');
    li.textContent = c;
    ul.appendChild(li);
  });
  host.append(h, ul);
}

async function ask(question) {
  const button = $('submit');
  button.disabled = true;
  runStages();
  const started = performance.now();

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, debug: true }),
    });
    const elapsed = Math.round(performance.now() - started);

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`${res.status}: ${detail.slice(0, 200)}`);
    }
    const body = await res.json();
    clearStages();

    const answerEl = $('answer');
    answerEl.textContent = body.answer;
    answerEl.dataset.refusal = body.answer === REFUSAL ? 'true' : 'false';

    if (body.follow_up_question) {
      const hint = document.createElement('span');
      hint.style.color = 'var(--muted)';
      hint.style.fontSize = '15px';
      hint.textContent = ' ' + body.follow_up_question;
      answerEl.appendChild(hint);
    }

    $('m-route').textContent = (body.route || '—').toUpperCase();
    $('m-decision').textContent = (body.decision || '—').toUpperCase();
    $('m-latency').textContent = elapsed + ' ms';

    const confidence = body.confidence ?? 0;
    $('gauge-fill').style.width = (confidence * 100) + '%';
    $('gauge-value').textContent = confidence.toFixed(3);

    renderCitations(body.citations);
    renderChunks(body.retrieved_chunks, body.citations);
  } catch (err) {
    clearStages();
    const answerEl = $('answer');
    answerEl.dataset.refusal = 'false';
    answerEl.textContent = 'Request failed: ' + err.message;
  } finally {
    button.disabled = false;
  }
}

$('form').addEventListener('submit', (event) => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (question) ask(question);
});

placeTicks();
loadHealth();
```

- [ ] **Step 4: Verify the page renders and answers**

Run: `uvicorn app.server:app --port 8000`

Open `http://127.0.0.1:8000/` and check:
- The health chip reads `OK · VECTOR+LEXICAL · GROQ` (or `DEGRADED` / `LEXICAL ONLY` if Task 4 is unresolved).
- Asking "What is QNH?" returns prose with at least one citation, and the telemetry rail fills in.
- Asking "What is the best chocolate cake recipe?" shows the refusal string in brick with a left rule.
- Cited chunks in the rail carry the amber left border; uncited ones do not.

- [ ] **Step 5: Run the anti-slop check**

Run:

```bash
grep -oE '#[0-9A-Fa-f]{6}' web/styles.css | tr 'a-f' 'A-F' | sort -u > /tmp/used.txt
grep -cE '#(6366F1|8B5CF6|9333EA|4F46E5|0F172A|020617|F59E0B|10B981|EF4444|FFFFFF|000000)' /tmp/used.txt
awk '{h=substr($0,2); if (substr(h,1,2)==substr(h,3,2) && substr(h,3,2)==substr(h,5,2)) print "NEUTRAL:", $0}' /tmp/used.txt
grep -cE 'border-radius: *[1-9]|linear-gradient|backdrop-filter|box-shadow' web/styles.css
```

Expected: first count `0`, no `NEUTRAL:` lines, last count `0`.

- [ ] **Step 6: Commit**

```bash
git add web/
git commit -m "feat: add Night Cockpit query console frontend"
```

---

### Task 17: Deployment, documentation, and final run

**Files:**
- Create: `Dockerfile`, `.dockerignore`
- Rewrite: `README.md`

**Interfaces:**
- Consumes: everything
- Produces: a runnable container and accurate documentation

- [ ] **Step 1: Write `.dockerignore`**

```
.git/
.venv/
venv/
__pycache__/
**/__pycache__/
*.pyc
data/
.env
.eval_cache/
.pytest_cache/
docs/
tests/
ground_truth_worksheet.txt
```

- [ ] **Step 2: Write `Dockerfile`**

The embedding model is baked in at build time so the container never depends on huggingface.co at runtime — the failure mode this whole plan started with.

```dockerfile
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

RUN adduser --disabled-password --gecos "" --uid 10001 airman

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image, then run fully offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && chown -R airman:airman /opt/hf

COPY app/ ./app/
COPY web/ ./web/
COPY vectorstore/ ./vectorstore/
COPY evaluate.py evaluation_set.json out_of_scope_set.json ./

ENV HF_LOCAL_FILES_ONLY=1
USER airman
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); \
sys.exit(0 if b['status']=='ok' else 1)"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build and run the container**

```bash
docker build -t airman-rag .
docker run --rm -p 8000:8000 -e GROQ_API_KEY="$GROQ_API_KEY" airman-rag
```

Expected: `/health` returns `"status": "ok"` with `vector_loaded: true`. Because the model is baked in, `HF_LOCAL_FILES_ONLY=1` is now safe inside the image.

- [ ] **Step 4: Rewrite `README.md`**

Sections, in order:
1. **What this is** — one paragraph, and the exact refusal string.
2. **Quick start** — POSIX *and* Windows activation lines (the current README gives only `.venv\Scripts\activate`).
3. **Configuration** — table of every `RAG_*` variable, default, and meaning. Call out explicitly that `HF_LOCAL_FILES_ONLY=1` requires the model to already be cached, and that setting it otherwise silently disabled semantic retrieval in the original version.
4. **Architecture** — the module map from the spec, with one line per module.
5. **API** — `/health`, `/ask`, `/ingest`, with real request/response examples captured from a running server.
6. **Evaluation** — how ground truth was authored, what each metric means, and a plain statement that the previous report's 100%/0% figures were an artifact of a circular metric.
7. **Frontend** — the Night Cockpit referent in two sentences, and the palette table.
8. **Deployment** — the Docker commands from Step 3.
9. **Testing** — `pip install -r requirements-dev.txt && pytest`.
10. **Known limitations** — no light theme; ground truth authored via retrieval; `data/` is untracked and must be supplied to re-ingest.

- [ ] **Step 5: Final verification**

```bash
pytest -q
python evaluate.py
head -40 report.md
git status --short
```

Expected: all tests pass; the report shows realistic (lower) figures with the methodology note; `git status` shows no `.env` and no `.pyc`.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore README.md report.md evaluation_detailed.csv
git commit -m "feat: add Dockerfile and rewrite README for the new architecture"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: module layout → 2–13; lazy init → 11; config object → 2; visible errors → 2, 4, 6, 8, 12; request flow → 11; Groq primary → 10, 11; ground truth → 14; metrics table → 15; judge model and fallback → 15; out-of-scope set → 14; frontend colour/type/shape/layout/motion/serving → 16; testing → every task; deployment → 17; hygiene → 1, 17; risks → documented in 17 Step 4 item 10.

**Deviation from the spec, recorded.** The spec said to migrate to `langchain-huggingface`'s `HuggingFaceEmbeddings`. Task 4 instead keeps `SentenceTransformerEmbeddings` and wraps it in `app/embeddings.py`. Reason: `langchain-huggingface` is not installed, and adding it risks disturbing the pinned `langchain-core==1.2.11`. Wrapping achieves the actual goal — loud failure and a dimension guard — without a new dependency. The deprecation warning is suppressed in `pytest.ini`.

**Scope addition beyond the spec.** Task 4 did not exist when the spec was written; the live FAISS outage was discovered during planning. It is now the highest-value task, and Tasks 15 and 16 both surface the degraded state rather than hiding it.

**Type consistency.** `RetrievedChunk`, `GenerationResult`, `GenerationOutcome`, `Decision`, `AskResult` are defined once in Task 3 and referenced unchanged after. `settings` is the trailing parameter on `is_grounded`, `clean_answer_text`, `refine_answer_clause`, `generate_extractive`, `generate_with_groq`, `judge`, and `LexicalIndex.search`. `Retriever.select_candidates` returns `(chunks, tokens)` in Tasks 8 and 11 alike. `engine.attach_index(vectorstore=..., lexical_index=...)` is keyword-identical in `conftest.py`, `test_api.py`, and `app/engine.py`.

**Placeholder scan.** No TBD/TODO. Every "move from graph.py" instruction names exact source line ranges and the exact mechanical substitutions. Task 14 Step 3 is authoring work rather than code, and carries a worked example plus explicit rules — it is the one task requiring human or agent judgement, by design.
