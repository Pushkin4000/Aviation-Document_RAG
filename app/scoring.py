import re
from typing import List

from app.models import RetrievedChunk

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


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]


def token_overlap_ratio(question_tokens: List[str], text: str) -> float:
    if not question_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    overlap = sum(1 for token in question_tokens if token in text_tokens)
    return overlap / max(1, len(question_tokens))


def normalize_for_match(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower())
    text = re.sub(r"[^a-z0-9 .,:;!?-]", "", text)
    return text.strip()


def noise_penalty(text: str) -> float:
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


def extract_acronyms(question: str) -> List[str]:
    return re.findall(r"\b[A-Z]{2,6}\b", question)


def definition_expansion_bonus(question: str, acronyms: List[str], text: str) -> float:
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


def is_definition_question(question: str) -> bool:
    q = question.lower().strip()
    if q.startswith(("what is", "what does", "define", "what are")):
        return True
    if re.match(r"how\s+is\s+.+\bdefined\b", q):
        return True
    return False


def calculate_confidence(question: str, chunks: List[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0

    top_chunks = chunks[:3]
    top_score = max(chunk.score for chunk in top_chunks)
    avg_score = sum(chunk.score for chunk in top_chunks) / len(top_chunks)
    top_overlap = max(chunk.lexical_overlap for chunk in top_chunks)
    avg_overlap = sum(chunk.lexical_overlap for chunk in top_chunks) / len(top_chunks)

    question_tokens = tokenize(question)
    merged_text = " ".join(chunk.document.page_content for chunk in top_chunks)
    coverage = token_overlap_ratio(question_tokens, merged_text)

    noise = sum(noise_penalty(chunk.document.page_content[:420]) for chunk in top_chunks) / len(top_chunks)

    confidence = (
        (0.38 * top_score)
        + (0.17 * avg_score)
        + (0.25 * top_overlap)
        + (0.10 * avg_overlap)
        + (0.20 * coverage)
        - (0.15 * noise)
    )
    return max(0.0, min(1.0, confidence))


def support_strength(
    question_tokens: List[str],
    answer: str,
    used_chunks: List[RetrievedChunk],
) -> float:
    if not used_chunks:
        return 0.0
    chunk_score = max(chunk.score for chunk in used_chunks)
    chunk_overlap = max(chunk.lexical_overlap for chunk in used_chunks)
    answer_question_overlap = token_overlap_ratio(question_tokens, answer)
    context_text = " ".join(chunk.document.page_content for chunk in used_chunks)
    answer_tokens = tokenize(answer)
    answer_context_overlap = token_overlap_ratio(answer_tokens, context_text)
    support = (
        (0.4 * chunk_score)
        + (0.25 * chunk_overlap)
        + (0.2 * answer_question_overlap)
        + (0.15 * answer_context_overlap)
    )
    return max(0.0, min(1.0, support))
