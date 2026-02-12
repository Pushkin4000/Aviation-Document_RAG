import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rapidfuzz import fuzz

load_dotenv()

try:
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
    ChatPromptTemplate = None
    JsonOutputParser = None

REFUSAL_MESSAGE = "This information is not available in the provided document(s)."
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = int(os.getenv("RAG_TOP_K", "8"))
MIN_RELEVANCE = float(os.getenv("RAG_MIN_RELEVANCE", "0.35"))
MIN_CHUNK_LEXICAL_OVERLAP = float(os.getenv("RAG_MIN_CHUNK_LEXICAL_OVERLAP", "0.12"))
MIN_SEGMENT_SCORE = float(os.getenv("RAG_MIN_SEGMENT_SCORE", "0.23"))
MIN_GROUNDED_SIMILARITY = float(os.getenv("RAG_MIN_GROUNDED_SIMILARITY", "0.65"))
MIN_GROUNDED_TOKEN_OVERLAP = float(os.getenv("RAG_MIN_GROUNDED_TOKEN_OVERLAP", "0.58"))
ANSWER_MAX_WORDS = int(os.getenv("RAG_ANSWER_MAX_WORDS", "65"))
GENERATION_MODE = os.getenv("RAG_GENERATION_MODE", "extractive").strip().lower()
GROQ_MODEL = os.getenv("RAG_GROQ_MODEL", "llama-3.3-70b-versatile")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1").strip() != "0"
ROUTER_MODE = os.getenv("RAG_ROUTER_MODE", "heuristic").strip().lower()
ROUTER_LLM_ENABLED = os.getenv("RAG_ROUTER_LLM_ENABLED", "0").strip() == "1"
MODEL_ROUTING_ENABLED = os.getenv("RAG_MODEL_ROUTING_ENABLED", "0").strip() == "1"
SIMPLE_MODEL = os.getenv("RAG_SIMPLE_MODEL", "llama-3.1-8b-instant")
COMPLEX_MODEL = os.getenv("RAG_COMPLEX_MODEL", "llama-3.3-70b-versatile")
CONFIDENCE_ANSWER_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_ANSWER_THRESHOLD", "0.48"))
CONFIDENCE_CLARIFY_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_CLARIFY_THRESHOLD", "0.34"))
LOW_CONFIDENCE_SUPPORT_THRESHOLD = float(os.getenv("RAG_LOW_CONFIDENCE_SUPPORT_THRESHOLD", "0.66"))

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "does",
    "do",
    "when",
    "where",
    "which",
    "who",
    "why",
    "stand",
    "with",
}


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
        except Exception:
            return 0

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", f"{self.source}:p{self.page}:c?"))


class AviationRAGEngine:
    def __init__(
        self,
        vectorstore_dir: str = VECTORSTORE_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = TOP_K,
        min_relevance: float = MIN_RELEVANCE,
    ) -> None:
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.embeddings: Optional[SentenceTransformerEmbeddings] = None
        self.vectorstore: Optional[FAISS] = None
        self.lexical_entries: List[Tuple[Document, set[str]]] = []
        self.refresh_index()

    def refresh_index(self) -> None:
        if not os.path.isdir(self.vectorstore_dir):
            self.vectorstore = None
            self.lexical_entries = []
            return
        self._load_lexical_entries()

        try:
            self.embeddings = SentenceTransformerEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"local_files_only": HF_LOCAL_FILES_ONLY},
            )
            self.vectorstore = FAISS.load_local(
                self.vectorstore_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            self.vectorstore = None

    def ask(self, question: str, debug: bool = False) -> Dict[str, object]:
        clean_question = question.strip()
        route = "simple"
        confidence = 0.0
        if not clean_question:
            return self._build_response(
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved=[],
                debug=debug,
                route=route,
                confidence=confidence,
                decision="refuse_empty_question",
                follow_up_question="Please ask a specific question from the provided aviation documents.",
            )

        retrieved = self._retrieve(clean_question)
        candidate_chunks, question_tokens = self._select_candidate_chunks(clean_question, retrieved)
        route = self._route_question(clean_question)
        confidence = self._calculate_confidence(clean_question, candidate_chunks)

        if not candidate_chunks:
            return self._build_response(
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved=retrieved,
                debug=debug,
                route=route,
                confidence=confidence,
                decision="refuse_no_relevant_chunks",
                follow_up_question=self._build_follow_up(clean_question, question_tokens, candidate_chunks),
            )

        if confidence < CONFIDENCE_CLARIFY_THRESHOLD:
            return self._build_response(
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved=retrieved,
                debug=debug,
                route=route,
                confidence=confidence,
                decision="refuse_low_confidence",
                follow_up_question=self._build_follow_up(clean_question, question_tokens, candidate_chunks),
            )

        if confidence < CONFIDENCE_ANSWER_THRESHOLD:
            salvage_answer, salvage_chunks = self._answer_extractive(
                question=clean_question,
                question_tokens=question_tokens,
                chunks=candidate_chunks,
            )
            if salvage_answer and salvage_chunks and self._is_grounded(salvage_answer, salvage_chunks):
                support = self._support_strength(question_tokens, salvage_answer, salvage_chunks)
                if support >= LOW_CONFIDENCE_SUPPORT_THRESHOLD:
                    return self._build_response(
                        answer=salvage_answer,
                        citations=self._unique_citations(salvage_chunks),
                        retrieved=retrieved,
                        debug=debug,
                        route=route,
                        confidence=confidence,
                        decision="answer_low_confidence_grounded",
                    )
            return self._build_response(
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved=retrieved,
                debug=debug,
                route=route,
                confidence=confidence,
                decision="clarify_low_confidence",
                follow_up_question=self._build_follow_up(clean_question, question_tokens, candidate_chunks),
            )

        answer, used_chunks = self._answer_from_context(
            question=clean_question,
            question_tokens=question_tokens,
            chunks=candidate_chunks,
            route=route,
        )
        if not answer or not used_chunks:
            return self._build_response(
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved=retrieved,
                debug=debug,
                route=route,
                confidence=confidence,
                decision="refuse_no_supported_answer",
                follow_up_question=self._build_follow_up(clean_question, question_tokens, candidate_chunks),
            )

        if not self._is_grounded(answer, used_chunks):
            return self._build_response(
                answer=REFUSAL_MESSAGE,
                citations=[],
                retrieved=retrieved,
                debug=debug,
                route=route,
                confidence=confidence,
                decision="refuse_grounding_failed",
                follow_up_question=self._build_follow_up(clean_question, question_tokens, candidate_chunks),
            )

        return self._build_response(
            answer=answer,
            citations=self._unique_citations(used_chunks),
            retrieved=retrieved,
            debug=debug,
            route=route,
            confidence=confidence,
            decision="answer",
        )

    def _retrieve(self, question: str) -> List[RetrievedChunk]:
        vector_results: List[RetrievedChunk] = []

        if self.vectorstore is not None:
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(question, k=self.top_k)
                vector_results = [
                    RetrievedChunk(document=doc, score=max(0.0, min(float(score or 0.0), 1.0)))
                    for doc, score in results
                ]
            except Exception:
                try:
                    docs = self.vectorstore.similarity_search(question, k=self.top_k)
                    vector_results = [RetrievedChunk(document=doc, score=0.0) for doc in docs]
                except Exception:
                    vector_results = []

        lexical_results = self._lexical_retrieve(question)
        if not vector_results:
            return lexical_results

        merged: Dict[str, RetrievedChunk] = {item.chunk_id: item for item in vector_results}
        for item in lexical_results:
            existing = merged.get(item.chunk_id)
            if existing is None or item.score > existing.score:
                merged[item.chunk_id] = item
        combined = list(merged.values())
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[: self.top_k]

    def _load_lexical_entries(self) -> None:
        index_path = Path(self.vectorstore_dir) / "index.pkl"
        if not index_path.exists():
            self.lexical_entries = []
            return

        try:
            with index_path.open("rb") as handle:
                docstore, _ = pickle.load(handle)
        except Exception:
            self.lexical_entries = []
            return

        raw_docs = []
        if hasattr(docstore, "_dict"):
            raw_docs = list(docstore._dict.values())
        self.lexical_entries = [
            (doc, set(self._tokenize(doc.page_content)))
            for doc in raw_docs
            if isinstance(doc, Document) and doc.page_content
        ]

    def _lexical_retrieve(self, question: str) -> List[RetrievedChunk]:
        if not self.lexical_entries:
            return []

        question_tokens = self._tokenize(question)
        if not question_tokens:
            return []
        qset = set(question_tokens)
        acronyms = self._extract_acronyms(question)
        asks_expansion = "stand for" in question.lower()

        scored: List[Tuple[float, float, Document]] = []
        for doc, token_set in self.lexical_entries:
            if not token_set:
                continue
            overlap = len(qset.intersection(token_set)) / max(1, len(qset))
            if overlap <= 0:
                continue
            fuzzy = fuzz.partial_ratio(question.lower(), doc.page_content[:900].lower()) / 100.0
            expansion_bonus = self._definition_expansion_bonus(
                question=question,
                acronyms=acronyms,
                text=doc.page_content[:1200],
            )
            noise_penalty = self._noise_penalty(doc.page_content[:500])
            score = min(1.0, max(0.0, (0.7 * overlap) + (0.2 * fuzzy) + expansion_bonus - noise_penalty))
            if asks_expansion and acronyms and expansion_bonus == 0.0:
                score *= 0.85
            if score < 0.05:
                continue
            scored.append((score, overlap, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(document=doc, score=score, lexical_overlap=overlap)
            for score, overlap, doc in scored[: self.top_k]
        ]

    def _select_candidate_chunks(
        self,
        question: str,
        retrieved: List[RetrievedChunk],
    ) -> Tuple[List[RetrievedChunk], List[str]]:
        question_tokens = self._tokenize(question)
        if not retrieved:
            return [], question_tokens

        candidates: List[RetrievedChunk] = []
        for idx, item in enumerate(retrieved):
            overlap = self._token_overlap_ratio(question_tokens, item.document.page_content)
            item.lexical_overlap = overlap
            if item.score >= self.min_relevance or overlap >= MIN_CHUNK_LEXICAL_OVERLAP or idx < 2:
                candidates.append(item)

        if not candidates:
            return [], question_tokens

        candidates.sort(
            key=lambda chunk: (
                (0.6 * chunk.score)
                + (0.35 * chunk.lexical_overlap)
                - (0.2 * self._noise_penalty(chunk.document.page_content[:420]))
            ),
            reverse=True,
        )
        return candidates[:5], question_tokens

    def _route_question(self, question: str) -> str:
        if ROUTER_MODE == "groq" and ROUTER_LLM_ENABLED and ChatGroq is not None and os.getenv("GROQ_API_KEY"):
            try:
                llm = ChatGroq(model=SIMPLE_MODEL, temperature=0)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "Classify the question complexity. Return JSON: {\"route\": \"simple\" | \"complex\"}.",
                        ),
                        ("human", "{question}"),
                    ]
                )
                chain = prompt | llm | JsonOutputParser()
                result = chain.invoke({"question": question})
                route = str(result.get("route", "simple")).strip().lower()
                if route in {"simple", "complex"}:
                    return route
            except Exception:
                pass
        return self._route_question_heuristic(question)

    def _route_question_heuristic(self, question: str) -> str:
        q = question.lower().strip()
        q_tokens = self._tokenize(question)
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
        if "?" in q and q.count("?") > 1:
            complexity += 0.06

        return "complex" if complexity >= 0.32 else "simple"

    def _calculate_confidence(self, question: str, chunks: List[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0

        top_chunks = chunks[:3]
        top_score = max(chunk.score for chunk in top_chunks)
        avg_score = sum(chunk.score for chunk in top_chunks) / len(top_chunks)
        top_overlap = max(chunk.lexical_overlap for chunk in top_chunks)
        avg_overlap = sum(chunk.lexical_overlap for chunk in top_chunks) / len(top_chunks)

        question_tokens = self._tokenize(question)
        merged_text = " ".join(chunk.document.page_content for chunk in top_chunks)
        coverage = self._token_overlap_ratio(question_tokens, merged_text)

        noise = sum(self._noise_penalty(chunk.document.page_content[:420]) for chunk in top_chunks) / len(top_chunks)

        confidence = (
            (0.38 * top_score)
            + (0.17 * avg_score)
            + (0.25 * top_overlap)
            + (0.10 * avg_overlap)
            + (0.20 * coverage)
            - (0.15 * noise)
        )
        return max(0.0, min(1.0, confidence))

    def _build_follow_up(
        self,
        question: str,
        question_tokens: List[str],
        chunks: List[RetrievedChunk],
    ) -> str:
        if not chunks:
            return "Please include the specific topic, procedure, or instrument from the provided documents."

        merged = " ".join(chunk.document.page_content for chunk in chunks[:2]).lower()
        missing = [token for token in question_tokens if token not in merged]
        hint_tokens = [token for token in missing if len(token) >= 4][:2]
        if hint_tokens:
            return (
                "Please clarify the exact context for "
                + ", ".join(hint_tokens)
                + " (for example: phase of flight, rule, or instrument)."
            )
        return "Please clarify the exact procedure or condition you want (for example: phase of flight, minima type, or regulation context)."

    def _answer_from_context(
        self,
        question: str,
        question_tokens: List[str],
        chunks: List[RetrievedChunk],
        route: str,
    ) -> Tuple[Optional[str], List[RetrievedChunk]]:
        if MODEL_ROUTING_ENABLED:
            model_name = SIMPLE_MODEL if route == "simple" else COMPLEX_MODEL
            llm_answer, llm_used = self._answer_with_groq_model(question, chunks, model_name)
            if llm_answer and llm_used:
                return llm_answer, llm_used
        elif GENERATION_MODE == "groq":
            llm_answer, llm_used = self._answer_with_groq_model(question, chunks, GROQ_MODEL)
            if llm_answer and llm_used:
                return llm_answer, llm_used

        return self._answer_extractive(question, question_tokens, chunks)

    def _answer_with_groq_model(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        model_name: str,
    ) -> Tuple[Optional[str], List[RetrievedChunk]]:
        if ChatGroq is None or not os.getenv("GROQ_API_KEY"):
            return None, []

        available_ids = {item.chunk_id for item in chunks}
        context_blocks = []
        for item in chunks[:5]:
            context_blocks.append(
                f"Chunk ID: {item.chunk_id}\n"
                f"Source: {item.source} (Page {item.page})\n"
                f"Content: {item.document.page_content}"
            )

        llm = ChatGroq(model=model_name, temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Answer only from the provided context. "
                        f"If context does not support the answer, output exactly: {REFUSAL_MESSAGE}\n"
                        "Return JSON with keys: answer (string), cited_chunk_ids (list of chunk IDs)."
                    ),
                ),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}",
                ),
            ]
        )
        chain = prompt | llm | JsonOutputParser()

        try:
            response = chain.invoke({"question": question, "context": "\n\n".join(context_blocks)})
        except Exception:
            return None, []

        answer = str(response.get("answer", "")).strip()
        cited_ids = [str(chunk_id) for chunk_id in response.get("cited_chunk_ids", [])]

        if answer == REFUSAL_MESSAGE:
            return answer, []

        valid_ids = [chunk_id for chunk_id in cited_ids if chunk_id in available_ids]
        if not valid_ids:
            return None, []

        used_chunks = [item for item in chunks if item.chunk_id in valid_ids]
        refined = self._refine_answer_clause(question, self._tokenize(question), answer)
        return self._clean_answer_text(refined), used_chunks

    def _answer_extractive(
        self,
        question: str,
        question_tokens: List[str],
        chunks: List[RetrievedChunk],
    ) -> Tuple[Optional[str], List[RetrievedChunk]]:
        candidates: List[Tuple[float, float, str, RetrievedChunk]] = []
        is_definition_question = self._is_definition_question(question)

        if is_definition_question:
            for chunk in chunks:
                acronym_probe = self._acronym_probe(question, chunk.document.page_content)
                if acronym_probe:
                    answer = self._clean_answer_text(acronym_probe)
                    if not self._is_definition_answer_form(question_tokens, answer):
                        continue
                    if len(answer) >= 20:
                        return answer, [chunk]

                probe = self._definition_probe(question_tokens, chunk.document.page_content)
                if probe:
                    answer = self._clean_answer_text(probe)
                    if not self._is_definition_answer_form(question_tokens, answer):
                        continue
                    if len(answer) >= 25:
                        return answer, [chunk]

        for chunk in chunks:
            for segment in self._segment_text(chunk.document.page_content):
                if is_definition_question:
                    if not self._is_definition_supportive(question, question_tokens, segment):
                        continue
                score, overlap = self._segment_score(question, question_tokens, segment)
                if score >= (MIN_SEGMENT_SCORE * 0.8) and overlap >= 0.08:
                    candidates.append((score, overlap, segment, chunk))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for score, overlap, segment, chunk in candidates:
                if score < MIN_SEGMENT_SCORE:
                    continue
                if overlap < 0.13:
                    continue
                refined = self._refine_answer_clause(question, question_tokens, segment)
                answer = self._clean_answer_text(refined)
                if is_definition_question and not self._is_definition_answer_form(question_tokens, answer):
                    continue
                if len(answer) >= 35:
                    return answer, [chunk]

        # Fallback: if relevant chunks exist but sentence segmentation is weak,
        # extract a dense keyword window instead of refusing.
        for chunk in chunks:
            window = self._keyword_window(question_tokens, chunk.document.page_content)
            if window:
                if is_definition_question:
                    if not self._is_definition_supportive(question, question_tokens, window):
                        continue
                refined = self._refine_answer_clause(question, question_tokens, window)
                answer = self._clean_answer_text(refined)
                if is_definition_question and not self._is_definition_answer_form(question_tokens, answer):
                    continue
                if len(answer) >= 35:
                    return answer, [chunk]

        return None, []

    def _acronym_probe(self, question: str, text: str) -> Optional[str]:
        acronyms = self._extract_acronyms(question)
        if not acronyms:
            return None
        asks_expansion = "stand for" in question.lower() or question.lower().startswith("what does")

        for acronym in acronyms:
            upper = acronym.upper()
            if asks_expansion:
                full_form_match = re.search(
                    rf"\b([A-Za-z][A-Za-z/-]*(?:[ -][A-Za-z][A-Za-z/-]*){{1,8}})\s*\(\s*{re.escape(upper)}\s*\)",
                    text,
                )
                if full_form_match:
                    full_form = re.sub(r"\s+", " ", full_form_match.group(1)).strip(" ,;:-")
                    words = re.findall(r"[A-Za-z][A-Za-z/-]*", full_form)
                    capitalized = sum(1 for word in words if word[0].isupper())
                    if len(words) >= 2 and capitalized >= max(2, len(words) // 2):
                        return f"{upper} stands for {full_form} ({upper})"

            definition_match = re.search(
                rf"\b{re.escape(upper)}\b\s+(?:is|means|refers to|stands for)\s+([^.?!]{{8,140}})",
                text,
                flags=re.IGNORECASE,
            )
            if definition_match:
                tail = definition_match.group(1).strip(" ,;:-")
                if tail:
                    if asks_expansion:
                        return f"{upper} stands for {tail}"
                    return f"{upper} is {tail}"

        return None

    def _definition_probe(self, question_tokens: List[str], text: str) -> Optional[str]:
        terms = [
            token
            for token in question_tokens
            if token
            not in {
                "defined",
                "definition",
                "mean",
                "means",
                "stand",
                "stands",
                "aviation",
                "flight",
                "mechanics",
                "meteorology",
            }
        ]
        if not terms:
            return None

        phrase_candidates: List[str] = []
        if len(terms) >= 2:
            phrase_candidates.append(" ".join(terms[:2]))
        if len(terms) >= 3:
            phrase_candidates.append(" ".join(terms[:3]))
        phrase_candidates.append(terms[0])

        seen = set()
        ordered_phrases: List[str] = []
        for phrase in phrase_candidates:
            if phrase in seen:
                continue
            seen.add(phrase)
            ordered_phrases.append(phrase)

        best_candidate = None
        best_score = -1.0
        for phrase in ordered_phrases:
            phrase_pattern = re.escape(phrase)
            label = phrase.capitalize()

            verb_match = re.search(
                rf"\b{phrase_pattern}\b\s*(?:is|means|defined as|refers to|stands for)\s+([^.?!]{{8,220}})",
                text,
                flags=re.IGNORECASE,
            )
            if verb_match:
                tail = verb_match.group(1).strip(" ,;:-")
                candidate = f"{label} is {tail}"
                score = 0.22 + (0.55 * self._token_overlap_ratio(question_tokens, candidate))
                score -= self._noise_penalty(candidate)
                if score > best_score and "?" not in candidate:
                    best_candidate = candidate
                    best_score = score

            glossary_match = re.search(
                rf"\b{phrase_pattern}\b\s*(?:[:\-]\s*)?(the|a|an)\s+([^.?!]{{8,200}})",
                text,
                flags=re.IGNORECASE,
            )
            if glossary_match:
                tail = f"{glossary_match.group(1)} {glossary_match.group(2)}".strip(" ,;:-")
                words = tail.split()
                if len(words) > 30:
                    tail = " ".join(words[:30])
                candidate = f"{label} is {tail}"
                score = 0.18 + (0.55 * self._token_overlap_ratio(question_tokens, candidate))
                score -= self._noise_penalty(candidate)
                if score > best_score and "?" not in candidate:
                    best_candidate = candidate
                    best_score = score

        return best_candidate

    def _is_definition_supportive(self, question: str, question_tokens: List[str], text: str) -> bool:
        if "?" in text:
            return False
        lower = text.lower()
        if re.search(r"\b[a-d]\.\s", lower):
            return False

        acronyms = self._extract_acronyms(question)
        if acronyms and self._definition_expansion_bonus(question, acronyms, text) >= 0.06:
            return True

        terms = [
            token
            for token in question_tokens
            if token not in {"defined", "definition", "mean", "means", "stand", "stands"}
        ]
        phrase_candidates: List[str] = []
        if len(terms) >= 2:
            phrase_candidates.append(" ".join(terms[:2]))
        if terms:
            phrase_candidates.append(terms[0])

        for phrase in phrase_candidates:
            phrase_pattern = re.escape(phrase)
            if re.search(
                rf"\b{phrase_pattern}\b\s*(?:is|means|defined as|refers to|stands for)\b",
                lower,
                flags=re.IGNORECASE,
            ):
                return True
            if re.search(
                rf"\b{phrase_pattern}\b\s*(?:[:\-]\s*)?(the|a|an)\b",
                lower,
                flags=re.IGNORECASE,
            ):
                return True
        return False

    def _is_definition_answer_form(self, question_tokens: List[str], answer: str) -> bool:
        lower = answer.lower()
        if self._noise_penalty(answer) >= 0.12:
            return False
        if re.search(r"\b[a-d]\.\s", lower):
            return False
        if re.search(r"\b(?:defined as|means|is)\s*[:\-]?\s*[a-d]\.?$", lower):
            return False
        if len(answer.split()) < 7:
            return False

        terms = [
            token
            for token in question_tokens
            if token not in {"defined", "definition", "mean", "means", "stand", "stands"}
        ]
        if not terms:
            return False

        anchor_phrases: List[str] = []
        if len(terms) >= 2:
            anchor_phrases.append(" ".join(terms[:2]))
        anchor_phrases.append(terms[0])
        has_anchor = any(phrase in lower for phrase in anchor_phrases)
        if not has_anchor:
            return False

        has_definition_cue = re.search(
            r"\b(is|means|defined as|refers to|stands for)\b",
            lower,
            flags=re.IGNORECASE,
        )
        return bool(has_definition_cue)

    def _segment_text(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return []

        segments: List[str] = []
        raw_segments = re.split(r"(?<=[.!?])\s+|(?<=:)\s+|(?<=;)\s+", normalized)
        for part in raw_segments:
            segment = part.strip()
            if self._is_valid_segment(segment):
                segments.append(segment)

        words = normalized.split()
        for start in range(0, len(words), 14):
            window = " ".join(words[start : start + 32]).strip()
            if self._is_valid_segment(window):
                segments.append(window)

        unique: List[str] = []
        for segment in segments:
            if any(fuzz.ratio(segment, seen) > 95 for seen in unique):
                continue
            unique.append(segment)
        return unique

    def _is_valid_segment(self, text: str) -> bool:
        if len(text) < 28 or len(text) > 320:
            return False
        words = text.split()
        if len(words) < 6:
            return False
        if self._noise_penalty(text) >= 0.2:
            return False
        alpha_ratio = sum(1 for char in text if char.isalpha()) / max(1, len(text))
        return alpha_ratio >= 0.55

    def _segment_score(self, question: str, question_tokens: List[str], segment: str) -> Tuple[float, float]:
        overlap = self._token_overlap_ratio(question_tokens, segment)
        fuzzy_ratio = fuzz.partial_ratio(question.lower(), segment.lower()) / 100.0

        definition_boost = 0.0
        if self._is_definition_question(question):
            has_definition_phrase = re.search(
                r"\b(is|means|defined as|defined|refers to|stands for)\b",
                segment.lower(),
            )
            if has_definition_phrase:
                definition_boost = 0.08
            definition_boost += self._definition_expansion_bonus(
                question=question,
                acronyms=self._extract_acronyms(question),
                text=segment,
            )
            if not has_definition_phrase:
                definition_boost -= 0.08

        reasoning_boost = 0.0
        if question.lower().startswith(("why", "how")):
            if re.search(r"\b(because|due to|therefore|as a result|results in)\b", segment.lower()):
                reasoning_boost = 0.05

        noise_penalty = self._noise_penalty(segment)
        score = (0.6 * overlap) + (0.3 * fuzzy_ratio) + definition_boost + reasoning_boost - noise_penalty
        return score, overlap

    def _keyword_window(self, question_tokens: List[str], text: str) -> Optional[str]:
        words = text.split()
        if not words or not question_tokens:
            return None

        lowered_words = [re.sub(r"[^a-z0-9]", "", word.lower()) for word in words]
        qset = set(question_tokens)

        best_window: Optional[List[str]] = None
        best_hits = 0
        best_density = 0.0

        window_size = 38
        stride = 8
        for start in range(0, max(1, len(words) - 3), stride):
            window_words = words[start : start + window_size]
            if len(window_words) < 10:
                continue
            window_norm = lowered_words[start : start + window_size]
            hits = len(qset.intersection(window_norm))
            if hits == 0:
                continue
            density = hits / max(1, len(set(window_norm)))
            if hits > best_hits or (hits == best_hits and density > best_density):
                best_hits = hits
                best_density = density
                best_window = window_words

        min_hits = 2 if len(question_tokens) >= 4 else 1
        if best_window is None or best_hits < min_hits:
            return None

        return " ".join(best_window)

    def _clean_answer_text(self, text: str) -> str:
        text = text.replace("Â", "")
        text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[A-Za-z]{2,8}\)\s*\d+\s*", "", text)
        text = re.sub(r"^[)\]-]+\s*", "", text)
        text = re.sub(r"^(?:\d+\s+){1,4}", "", text)
        text = re.sub(r"^([A-Za-z]{3,20})\s+\d+\s+\1\s+", "", text)
        text = re.sub(r"^[A-Za-z]{3,20}\s+\d+\s*[A-Za-z]{3,20}\s+", "", text)
        text = re.sub(r"^([A-Za-z]{3,20})\s+\1\s+", "", text)
        text = re.sub(r"^(?:Questions?\s*)+", "", text, flags=re.IGNORECASE)
        if len(re.findall(r"\b[a-d]\.\s", text.lower())) >= 2:
            text = re.split(r"\b[a-d]\.\s", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        if len(words) > ANSWER_MAX_WORDS:
            text = " ".join(words[:ANSWER_MAX_WORDS]).rstrip(" ,;:")

        text = text.strip(" ,;:")
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _refine_answer_clause(self, question: str, question_tokens: List[str], text: str) -> str:
        clauses = re.split(r"(?<=[.!?;:])\s+|\?\s+|!\s+", text)
        acronyms = self._extract_acronyms(question)
        best_clause = text
        best_score = -1.0

        for clause in clauses:
            candidate = clause.strip(" -")
            if len(candidate) < 18:
                continue
            overlap = self._token_overlap_ratio(question_tokens, candidate)
            score = overlap - self._noise_penalty(candidate)
            if self._is_definition_question(question):
                score += self._definition_expansion_bonus(question, acronyms, candidate)
                if re.search(r"\b(is|means|defined as|refers to|stands for)\b", candidate.lower()):
                    score += 0.08
            if score > best_score:
                best_score = score
                best_clause = candidate

        if best_score >= 0.12:
            return best_clause
        return text

    def _is_grounded(self, answer: str, used_chunks: List[RetrievedChunk]) -> bool:
        if answer == REFUSAL_MESSAGE:
            return True

        lower_answer = answer.lower()
        stripped = lower_answer.strip()
        if stripped.startswith(("what is", "which is", "what does", "why ", "how ")):
            return False
        if re.search(r"\b[a-d]\.\s", lower_answer):
            return False
        if lower_answer.count("?") > 0:
            return False
        if self._noise_penalty(answer) >= 0.14:
            return False

        alpha_chars = sum(1 for char in answer if char.isalpha())
        if (alpha_chars / max(1, len(answer))) < 0.55:
            return False

        normalized_answer = self._normalize_for_match(answer)
        normalized_context = self._normalize_for_match(
            " ".join(chunk.document.page_content for chunk in used_chunks)
        )

        if not normalized_context.strip():
            return False

        if normalized_answer and normalized_answer in normalized_context:
            return True

        clauses = [piece.strip() for piece in re.split(r"[.;]", normalized_answer) if len(piece.strip()) >= 20]
        if clauses and any(clause in normalized_context for clause in clauses):
            return True

        similarity = fuzz.partial_ratio(normalized_answer, normalized_context) / 100.0
        answer_token_list = self._tokenize(answer)
        if not answer_token_list:
            return False
        answer_tokens = set(answer_token_list)
        context_tokens = set(self._tokenize(normalized_context))
        overlap = len(answer_tokens.intersection(context_tokens)) / max(1, len(answer_tokens))
        required_overlap = MIN_GROUNDED_TOKEN_OVERLAP
        if len(answer_token_list) <= 10:
            required_overlap = max(0.5, MIN_GROUNDED_TOKEN_OVERLAP - 0.06)
        return similarity >= MIN_GROUNDED_SIMILARITY and overlap >= required_overlap

    def _support_strength(
        self,
        question_tokens: List[str],
        answer: str,
        used_chunks: List[RetrievedChunk],
    ) -> float:
        if not used_chunks:
            return 0.0
        chunk_score = max(chunk.score for chunk in used_chunks)
        chunk_overlap = max(chunk.lexical_overlap for chunk in used_chunks)
        answer_question_overlap = self._token_overlap_ratio(question_tokens, answer)
        context_text = " ".join(chunk.document.page_content for chunk in used_chunks)
        answer_tokens = self._tokenize(answer)
        answer_context_overlap = self._token_overlap_ratio(answer_tokens, context_text)
        support = (
            (0.4 * chunk_score)
            + (0.25 * chunk_overlap)
            + (0.2 * answer_question_overlap)
            + (0.15 * answer_context_overlap)
        )
        return max(0.0, min(1.0, support))

    def _build_response(
        self,
        answer: str,
        citations: List[str],
        retrieved: List[RetrievedChunk],
        debug: bool,
        route: str,
        confidence: float,
        decision: str,
        follow_up_question: Optional[str] = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "answer": answer,
            "citations": citations,
            "route": route,
            "confidence": round(confidence, 4),
            "decision": decision,
        }
        if follow_up_question:
            payload["follow_up_question"] = follow_up_question
        if debug:
            payload["retrieved_chunks"] = [
                {
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "page": item.page,
                    "score": round(item.score, 4),
                    "lexical_overlap": round(item.lexical_overlap, 4),
                    "content_snippet": item.document.page_content[:320],
                }
                for item in retrieved[: max(3, min(len(retrieved), self.top_k))]
            ]
        return payload

    def _unique_citations(self, chunks: List[RetrievedChunk]) -> List[str]:
        seen = set()
        citations = []
        for chunk in chunks:
            citation = f"{chunk.source} (Page {chunk.page})"
            if citation in seen:
                continue
            seen.add(citation)
            citations.append(citation)
        return citations

    def _extract_acronyms(self, question: str) -> List[str]:
        return re.findall(r"\b[A-Z]{2,6}\b", question)

    def _definition_expansion_bonus(self, question: str, acronyms: List[str], text: str) -> float:
        if not acronyms:
            return 0.0

        lower_text = text.lower()
        bonus = 0.0
        asks_expansion = "stand for" in question.lower() or question.lower().startswith("what does")

        for acronym in acronyms:
            upper = acronym.upper()
            pattern_paren = rf"\([ ]*{re.escape(upper)}[ ]*\)"
            pattern_full = rf"\b[A-Za-z][A-Za-z -]{{3,80}}{pattern_paren}"
            pattern_is = rf"\b{re.escape(upper)}\b\s+(is|means|refers to|stands for)\b"
            if re.search(pattern_full, text):
                bonus += 0.12 if asks_expansion else 0.08
            if re.search(pattern_is, lower_text, flags=re.IGNORECASE):
                bonus += 0.08
            if re.search(rf"\b{re.escape(upper.lower())}\b", lower_text):
                bonus += 0.02
        return min(bonus, 0.2)

    def _noise_penalty(self, text: str) -> float:
        lower = text.lower()
        penalty = 0.0
        if "questions" in lower:
            penalty += 0.12
        if re.search(r"\b[a-d]\.\s", lower):
            penalty += 0.08
        if lower.count("?") >= 2:
            penalty += 0.06
        if "figure " in lower and " is " not in lower:
            penalty += 0.04
        return min(penalty, 0.22)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]

    def _token_overlap_ratio(self, question_tokens: List[str], text: str) -> float:
        if not question_tokens:
            return 0.0
        text_tokens = set(self._tokenize(text))
        overlap = sum(1 for token in question_tokens if token in text_tokens)
        return overlap / max(1, len(question_tokens))

    def _normalize_for_match(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.lower())
        text = re.sub(r"[^a-z0-9 .,:;!?-]", "", text)
        return text.strip()

    def _is_definition_question(self, question: str) -> bool:
        q = question.lower().strip()
        if q.startswith(("what is", "what does", "define", "what are")):
            return True
        if re.match(r"how\s+is\s+.+\bdefined\b", q):
            return True
        return False


rag_engine = AviationRAGEngine()
