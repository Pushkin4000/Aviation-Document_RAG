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

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """Public access to raw retrieval, for evaluation and tooling."""
        return self._retriever.retrieve(question)

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
