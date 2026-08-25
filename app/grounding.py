import re
from typing import List

from rapidfuzz import fuzz

from app import scoring
from app.config import Settings
from app.logging_setup import get_logger
from app.models import REFUSAL_MESSAGE, RetrievedChunk

logger = get_logger("grounding")


def is_grounded(
    answer: str, used_chunks: List[RetrievedChunk], settings: Settings, abstractive: bool = False
) -> bool:
    if answer == REFUSAL_MESSAGE:
        return True

    lower_answer = answer.lower()
    stripped = lower_answer.strip()
    if stripped.startswith(("what is", "which is", "what does", "why ", "how ")):
        logger.debug("grounding rejected: question-shaped answer")
        return False
    if re.search(r"\b[a-d]\.\s", lower_answer):
        logger.debug("grounding rejected: multiple-choice residue")
        return False
    if lower_answer.count("?") > 0:
        logger.debug("grounding rejected: question mark present")
        return False
    if scoring.noise_penalty(answer) >= 0.14:
        logger.debug("grounding rejected: noise penalty too high")
        return False

    alpha_chars = sum(1 for char in answer if char.isalpha())
    if (alpha_chars / max(1, len(answer))) < 0.55:
        logger.debug("grounding rejected: alpha ratio too low")
        return False

    normalized_answer = scoring.normalize_for_match(answer)
    normalized_context = scoring.normalize_for_match(
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
    answer_token_list = scoring.tokenize(answer)
    if not answer_token_list:
        return False
    answer_tokens = set(answer_token_list)
    context_tokens = set(scoring.tokenize(normalized_context))
    overlap = len(answer_tokens.intersection(context_tokens)) / max(1, len(answer_tokens))
    if abstractive:
        min_similarity = settings.min_grounded_similarity_abstractive
        min_overlap = settings.min_grounded_token_overlap_abstractive
    else:
        min_similarity = settings.min_grounded_similarity
        min_overlap = settings.min_grounded_token_overlap
    required_overlap = min_overlap
    if len(answer_token_list) <= 10:
        required_overlap = max(0.5, min_overlap - 0.06)
    return similarity >= min_similarity and overlap >= required_overlap
