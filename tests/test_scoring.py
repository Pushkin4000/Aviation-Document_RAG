from langchain_core.documents import Document

from app.models import RetrievedChunk
from app.scoring import (
    calculate_confidence,
    extract_acronyms,
    is_definition_question,
    noise_penalty,
    normalize_for_match,
    support_strength,
    token_overlap_ratio,
    tokenize,
)


def _chunk(text, score=0.8, overlap=0.5):
    return RetrievedChunk(
        document=Document(page_content=text, metadata={"source": "s.pdf", "page": 1}),
        score=score,
        lexical_overlap=overlap,
    )


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("What is the QNH of a runway?") == ["qnh", "runway"]


def test_tokenize_is_case_and_punctuation_insensitive():
    assert tokenize("COLD-FRONT, cold front!") == ["cold", "front", "cold", "front"]


def test_token_overlap_ratio_bounds():
    assert token_overlap_ratio([], "anything") == 0.0
    assert token_overlap_ratio(["qnh"], "the qnh setting") == 1.0
    assert token_overlap_ratio(["qnh", "runway"], "the qnh setting") == 0.5


def test_noise_penalty_flags_exam_style_text():
    clean = "A cold front occurs when cold air replaces warm air."
    exam = "Questions a. one b. two c. three d. four? which is correct?"
    assert noise_penalty(clean) == 0.0
    assert noise_penalty(exam) > 0.2 - 1e-9
    assert noise_penalty(exam) <= 0.22


def test_extract_acronyms():
    assert extract_acronyms("What does VOR and DME mean?") == ["VOR", "DME"]
    assert extract_acronyms("what is a cold front") == []


def test_is_definition_question():
    assert is_definition_question("What is QNH?")
    assert is_definition_question("How is dew point defined?")
    assert not is_definition_question("Why does icing form on the wing?")


def test_confidence_is_zero_without_chunks():
    assert calculate_confidence("anything", []) == 0.0


def test_confidence_is_bounded_and_ordered():
    strong = [_chunk("qnh is the altimeter setting at mean sea level", 0.95, 0.9)]
    weak = [_chunk("unrelated text about catering menus", 0.05, 0.0)]
    hi = calculate_confidence("What is QNH?", strong)
    lo = calculate_confidence("What is QNH?", weak)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi > lo


def test_support_strength_bounded():
    c = _chunk("a cold front replaces warm air", 0.9, 0.8)
    v = support_strength(["cold", "front"], "A cold front replaces warm air.", [c])
    assert 0.0 <= v <= 1.0
    assert support_strength(["x"], "y", []) == 0.0


def test_normalize_for_match_strips_symbols():
    assert normalize_for_match("  QNH:  1013 hPa!! ") == "qnh: 1013 hpa!!"
