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


def _cache_key(question: str, answer: str, expected: str, discriminator: str) -> str:
    # MIN-4: the discriminator (which judge path this call would take, and
    # which model) must be part of the key. Without it, a cached
    # lexical-fallback verdict from a no-key run silently keeps being
    # returned after a valid GROQ_API_KEY is supplied -- the re-run looks
    # like it used the LLM judge but is actually replaying stale
    # lexical-fallback output.
    payload = f"{question}|{answer}|{expected}|{discriminator}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


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
    discriminator = f"llm:{settings.judge_model}" if settings.groq_api_key else "lexical-fallback"
    key = _cache_key(question, answer, expected_answer or "", discriminator)
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
