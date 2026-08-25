from langchain_core.documents import Document

from app.config import Settings
from app.generation.extractive import clean_answer_text, generate_extractive
from app.models import GenerationOutcome, RetrievedChunk

SETTINGS = Settings(_env_file=None)


def _chunk(text):
    return RetrievedChunk(
        document=Document(page_content=text, metadata={"source": "s.pdf", "page": 1, "chunk_id": "s.pdf:p1:c1"}),
        score=0.9,
        lexical_overlap=0.8,
    )


def test_extracts_a_definition():
    chunks = [_chunk("QNH is the altimeter subscale setting to obtain elevation above mean sea level when on the ground.")]
    result = generate_extractive("What is QNH?", ["qnh"], chunks, SETTINGS)
    assert result.outcome is GenerationOutcome.ANSWERED
    assert "qnh" in result.answer.lower()
    assert result.chunks


def test_returns_unavailable_when_no_chunks():
    result = generate_extractive("What is QNH?", ["qnh"], [], SETTINGS)
    assert result.outcome is GenerationOutcome.UNAVAILABLE
    assert result.answer is None


def test_returns_unavailable_when_nothing_matches():
    chunks = [_chunk("Catering arrangements for long haul cabin service on wide body aircraft.")]
    result = generate_extractive("What is the boiling point of water?", ["boiling", "point", "water"], chunks, SETTINGS)
    assert result.outcome is GenerationOutcome.UNAVAILABLE


def test_clean_answer_truncates_to_max_words():
    settings = Settings(_env_file=None, answer_max_words=10)
    out = clean_answer_text(
        "one two three four five six seven eight nine ten eleven twelve", settings
    )
    assert len(out.rstrip(".").split()) == 10


def test_clean_answer_terminates_sentence():
    assert clean_answer_text("a grounded statement", SETTINGS).endswith(".")


def test_clean_answer_cuts_multiple_choice_tail():
    out = clean_answer_text("The front is cold a. warm b. cold c. occluded", SETTINGS)
    assert "b." not in out
