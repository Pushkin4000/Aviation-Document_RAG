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
