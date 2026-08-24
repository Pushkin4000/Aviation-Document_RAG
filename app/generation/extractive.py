import re
from typing import List, Optional, Tuple

from rapidfuzz import fuzz

from app.config import Settings
from app.models import GenerationResult, RetrievedChunk
from app.scoring import (
    definition_expansion_bonus,
    extract_acronyms,
    is_definition_question,
    noise_penalty,
    token_overlap_ratio,
)


def generate_extractive(
    question: str,
    question_tokens: List[str],
    chunks: List[RetrievedChunk],
    settings: Settings,
) -> GenerationResult:
    if not chunks:
        return GenerationResult.unavailable()

    candidates: List[Tuple[float, float, str, RetrievedChunk]] = []
    definition_question = is_definition_question(question)

    if definition_question:
        for chunk in chunks:
            acronym_probe = _acronym_probe(question, chunk.document.page_content)
            if acronym_probe:
                answer = clean_answer_text(acronym_probe, settings)
                if not _is_definition_answer_form(question_tokens, answer):
                    continue
                if len(answer) >= 20:
                    return GenerationResult.answered(answer, [chunk])

            probe = _definition_probe(question_tokens, chunk.document.page_content)
            if probe:
                answer = clean_answer_text(probe, settings)
                if not _is_definition_answer_form(question_tokens, answer):
                    continue
                if len(answer) >= 25:
                    return GenerationResult.answered(answer, [chunk])

    for chunk in chunks:
        for segment in _segment_text(chunk.document.page_content):
            if definition_question:
                if not _is_definition_supportive(question, question_tokens, segment):
                    continue
            score, overlap = _segment_score(question, question_tokens, segment, settings)
            if score >= (settings.min_segment_score * 0.8) and overlap >= 0.08:
                candidates.append((score, overlap, segment, chunk))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for score, overlap, segment, chunk in candidates:
            if score < settings.min_segment_score:
                continue
            if overlap < 0.13:
                continue
            refined = refine_answer_clause(question, question_tokens, segment)
            answer = clean_answer_text(refined, settings)
            if definition_question and not _is_definition_answer_form(question_tokens, answer):
                continue
            if len(answer) >= 35:
                return GenerationResult.answered(answer, [chunk])

    # Fallback: if relevant chunks exist but sentence segmentation is weak,
    # extract a dense keyword window instead of refusing.
    for chunk in chunks:
        window = _keyword_window(question_tokens, chunk.document.page_content)
        if window:
            if definition_question:
                if not _is_definition_supportive(question, question_tokens, window):
                    continue
            refined = refine_answer_clause(question, question_tokens, window)
            answer = clean_answer_text(refined, settings)
            if definition_question and not _is_definition_answer_form(question_tokens, answer):
                continue
            if len(answer) >= 35:
                return GenerationResult.answered(answer, [chunk])

    return GenerationResult.unavailable()


def _acronym_probe(question: str, text: str) -> Optional[str]:
    acronyms = extract_acronyms(question)
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


def _definition_probe(question_tokens: List[str], text: str) -> Optional[str]:
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
            score = 0.22 + (0.55 * token_overlap_ratio(question_tokens, candidate))
            score -= noise_penalty(candidate)
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
            score = 0.18 + (0.55 * token_overlap_ratio(question_tokens, candidate))
            score -= noise_penalty(candidate)
            if score > best_score and "?" not in candidate:
                best_candidate = candidate
                best_score = score

    return best_candidate


def _is_definition_supportive(question: str, question_tokens: List[str], text: str) -> bool:
    if "?" in text:
        return False
    lower = text.lower()
    if re.search(r"\b[a-d]\.\s", lower):
        return False

    acronyms = extract_acronyms(question)
    if acronyms and definition_expansion_bonus(question, acronyms, text) >= 0.06:
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


def _is_definition_answer_form(question_tokens: List[str], answer: str) -> bool:
    lower = answer.lower()
    if noise_penalty(answer) >= 0.12:
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


def _segment_text(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []

    segments: List[str] = []
    raw_segments = re.split(r"(?<=[.!?])\s+|(?<=:)\s+|(?<=;)\s+", normalized)
    for part in raw_segments:
        segment = part.strip()
        if _is_valid_segment(segment):
            segments.append(segment)

    words = normalized.split()
    for start in range(0, len(words), 14):
        window = " ".join(words[start : start + 32]).strip()
        if _is_valid_segment(window):
            segments.append(window)

    unique: List[str] = []
    for segment in segments:
        if any(fuzz.ratio(segment, seen) > 95 for seen in unique):
            continue
        unique.append(segment)
    return unique


def _is_valid_segment(text: str) -> bool:
    if len(text) < 28 or len(text) > 320:
        return False
    words = text.split()
    if len(words) < 6:
        return False
    if noise_penalty(text) >= 0.2:
        return False
    alpha_ratio = sum(1 for char in text if char.isalpha()) / max(1, len(text))
    return alpha_ratio >= 0.55


def _segment_score(
    question: str,
    question_tokens: List[str],
    segment: str,
    settings: Settings,
) -> Tuple[float, float]:
    overlap = token_overlap_ratio(question_tokens, segment)
    fuzzy_ratio = fuzz.partial_ratio(question.lower(), segment.lower()) / 100.0

    definition_boost = 0.0
    if is_definition_question(question):
        has_definition_phrase = re.search(
            r"\b(is|means|defined as|defined|refers to|stands for)\b",
            segment.lower(),
        )
        if has_definition_phrase:
            definition_boost = 0.08
        definition_boost += definition_expansion_bonus(
            question=question,
            acronyms=extract_acronyms(question),
            text=segment,
        )
        if not has_definition_phrase:
            definition_boost -= 0.08

    reasoning_boost = 0.0
    if question.lower().startswith(("why", "how")):
        if re.search(r"\b(because|due to|therefore|as a result|results in)\b", segment.lower()):
            reasoning_boost = 0.05

    penalty = noise_penalty(segment)
    score = (0.6 * overlap) + (0.3 * fuzzy_ratio) + definition_boost + reasoning_boost - penalty
    return score, overlap


def _keyword_window(question_tokens: List[str], text: str) -> Optional[str]:
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


def clean_answer_text(text: str, settings: Settings) -> str:
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
    if len(words) > settings.answer_max_words:
        text = " ".join(words[: settings.answer_max_words]).rstrip(" ,;:")

    text = text.strip(" ,;:")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def refine_answer_clause(question: str, question_tokens: List[str], text: str) -> str:
    clauses = re.split(r"(?<=[.!?;:])\s+|\?\s+|!\s+", text)
    acronyms = extract_acronyms(question)
    best_clause = text
    best_score = -1.0

    for clause in clauses:
        candidate = clause.strip(" -")
        if len(candidate) < 18:
            continue
        overlap = token_overlap_ratio(question_tokens, candidate)
        score = overlap - noise_penalty(candidate)
        if is_definition_question(question):
            score += definition_expansion_bonus(question, acronyms, candidate)
            if re.search(r"\b(is|means|defined as|refers to|stands for)\b", candidate.lower()):
                score += 0.08
        if score > best_score:
            best_score = score
            best_clause = candidate

    if best_score >= 0.12:
        return best_clause
    return text
