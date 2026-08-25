import time
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


def _retry_after_seconds(exc: Exception) -> float:
    """Seconds the provider asked us to wait, or 0.0 if it did not say."""
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        raw = headers.get(header)
        if raw is None:
            continue
        try:
            return float(str(raw).rstrip("s"))
        except ValueError:
            continue
    return 0.0


def _is_rate_limit(exc: Exception) -> bool:
    if type(exc).__name__ == "RateLimitError":
        return True
    return getattr(getattr(exc, "response", None), "status_code", None) == 429


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

    context = _build_context(chunks)
    for attempt in range(settings.groq_max_retries + 1):
        try:
            response = _invoke_chain(question, context, model_name, settings)
            break
        except Exception as exc:
            last_attempt = attempt == settings.groq_max_retries
            if last_attempt or not _is_rate_limit(exc):
                logger.warning("Groq generation failed (%s: %s)", type(exc).__name__, exc)
                return GenerationResult.unavailable()
            # Back off only for a wait we are willing to sit through. The
            # daily-cap 429 asks for minutes; that is not a blip, so fall
            # back to extraction immediately instead of stalling the caller.
            wait = _retry_after_seconds(exc) or (2.0 ** attempt)
            if wait > settings.groq_retry_max_wait_seconds:
                logger.warning(
                    "Groq rate limit needs %.0fs (> %.0fs cap), falling back without retry",
                    wait,
                    settings.groq_retry_max_wait_seconds,
                )
                return GenerationResult.unavailable()
            logger.info("Groq rate limited, retrying in %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)

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

    refined = refine_answer_clause(question, tokenize(question), answer)
    return GenerationResult.answered(clean_answer_text(refined, settings), used)
