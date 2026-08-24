"""Hybrid (vector + lexical) retrieval, extracted from the original engine.

Fixes bug #2: `_select_candidate_chunks` used to admit the top two retrieved
chunks unconditionally (`or idx < 2`), regardless of score or lexical
overlap. That meant the candidate list was never empty and
`Decision.REFUSE_NO_RELEVANT_CHUNKS` could never fire. `select_candidates`
below admits chunks on evidence alone.
"""

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from rapidfuzz import fuzz

from app import scoring
from app.config import Settings
from app.logging_setup import get_logger
from app.models import RetrievedChunk

logger = get_logger("retrieval")


class LexicalIndex:
    """Keyword/fuzzy fallback index built from a plain list of Documents."""

    def __init__(self, entries: List[Document]):
        self.entries: List[Tuple[Document, set]] = [
            (doc, set(scoring.tokenize(doc.page_content)))
            for doc in entries
            if isinstance(doc, Document) and doc.page_content
        ]

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_docstore(cls, docstore) -> "LexicalIndex":
        """Build from a loaded FAISS docstore instead of unpickling index.pkl by hand."""
        raw = list(getattr(docstore, "_dict", {}).values())
        return cls([d for d in raw if isinstance(d, Document) and d.page_content])

    def search(self, question: str, top_k: int, settings: Settings) -> List[RetrievedChunk]:
        if not self.entries:
            return []

        question_tokens = scoring.tokenize(question)
        if not question_tokens:
            return []
        qset = set(question_tokens)
        acronyms = scoring.extract_acronyms(question)
        asks_expansion = "stand for" in question.lower()

        scored: List[Tuple[float, float, Document]] = []
        for doc, token_set in self.entries:
            if not token_set:
                continue
            overlap = len(qset.intersection(token_set)) / max(1, len(qset))
            if overlap <= 0:
                continue
            fuzzy = fuzz.partial_ratio(question.lower(), doc.page_content[:900].lower()) / 100.0
            expansion_bonus = scoring.definition_expansion_bonus(
                question=question,
                acronyms=acronyms,
                text=doc.page_content[:1200],
            )
            noise = scoring.noise_penalty(doc.page_content[:500])
            score = min(1.0, max(0.0, (0.7 * overlap) + (0.2 * fuzzy) + expansion_bonus - noise))
            if asks_expansion and acronyms and expansion_bonus == 0.0:
                score *= 0.85
            if score < 0.05:
                continue
            scored.append((score, overlap, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(document=doc, score=score, lexical_overlap=overlap)
            for score, overlap, doc in scored[:top_k]
        ]


class Retriever:
    """Hybrid retriever: FAISS vector search merged with the lexical fallback."""

    def __init__(self, vectorstore, lexical_index: LexicalIndex, settings: Settings):
        self.vectorstore = vectorstore
        self.lexical_index = lexical_index
        self.settings = settings

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        vector_results: List[RetrievedChunk] = []

        if self.vectorstore is not None:
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    question, k=self.settings.top_k
                )
                vector_results = [
                    RetrievedChunk(document=doc, score=max(0.0, min(float(score or 0.0), 1.0)))
                    for doc, score in results
                ]
            except Exception as exc:
                logger.warning("Vector search failed (%s: %s), using lexical only", type(exc).__name__, exc)
                vector_results = []

        lexical_results = self.lexical_index.search(question, self.settings.top_k, self.settings)
        if not vector_results:
            return lexical_results

        merged: Dict[str, RetrievedChunk] = {item.chunk_id: item for item in vector_results}
        for item in lexical_results:
            existing = merged.get(item.chunk_id)
            if existing is None or item.score > existing.score:
                merged[item.chunk_id] = item
        combined = list(merged.values())
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[: self.settings.top_k]

    def select_candidates(
        self,
        question: str,
        retrieved: List[RetrievedChunk],
    ) -> Tuple[List[RetrievedChunk], List[str]]:
        question_tokens = scoring.tokenize(question)
        if not retrieved:
            return [], question_tokens

        candidates: List[RetrievedChunk] = []
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
