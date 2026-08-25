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
            "content": self.document.page_content,
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
