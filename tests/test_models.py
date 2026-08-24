from langchain_core.documents import Document

from app.models import (
    REFUSAL_MESSAGE,
    AskResult,
    Decision,
    GenerationOutcome,
    GenerationResult,
    RetrievedChunk,
)


def _chunk(text="Cold fronts replace warm air.", page=3, source="met.pdf", score=0.8):
    return RetrievedChunk(
        document=Document(
            page_content=text,
            metadata={"source": source, "page": page, "chunk_id": f"{source}:p{page}:c1"},
        ),
        score=score,
    )


def test_refusal_message_is_exact():
    assert REFUSAL_MESSAGE == "This information is not available in the provided document(s)."


def test_chunk_exposes_metadata():
    c = _chunk()
    assert c.source == "met.pdf"
    assert c.page == 3
    assert c.chunk_id == "met.pdf:p3:c1"


def test_chunk_tolerates_missing_metadata():
    c = RetrievedChunk(document=Document(page_content="x", metadata={}), score=0.1)
    assert c.source == "Unknown"
    assert c.page == 0


def test_declined_is_distinguishable_from_unavailable():
    declined = GenerationResult.declined()
    unavailable = GenerationResult.unavailable()
    assert declined.outcome is GenerationOutcome.DECLINED
    assert unavailable.outcome is GenerationOutcome.UNAVAILABLE
    assert declined.outcome is not unavailable.outcome
    # Both carry no chunks; the outcome is the only signal. This is the
    # distinction the old tuple-return collapsed, causing bug #1.
    assert declined.chunks == [] and unavailable.chunks == []


def test_answered_carries_answer_and_chunks():
    c = _chunk()
    r = GenerationResult.answered("A cold front replaces warm air.", [c])
    assert r.outcome is GenerationOutcome.ANSWERED
    assert r.chunks == [c]


def test_payload_omits_retrieved_chunks_unless_debug():
    result = AskResult(
        answer="x.",
        citations=["met.pdf (Page 3)"],
        route="simple",
        confidence=0.7123456,
        decision=Decision.ANSWER,
        retrieved=[_chunk()],
    )
    assert "retrieved_chunks" not in result.to_payload(debug=False, top_k=8)
    payload = result.to_payload(debug=True, top_k=8)
    assert payload["retrieved_chunks"][0]["chunk_id"] == "met.pdf:p3:c1"
    assert payload["confidence"] == 0.7123
    assert payload["decision"] == "answer"


def test_payload_omits_empty_follow_up():
    result = AskResult(
        answer=REFUSAL_MESSAGE,
        citations=[],
        route="simple",
        confidence=0.1,
        decision=Decision.REFUSE_LOW_CONFIDENCE,
        follow_up_question=None,
    )
    assert "follow_up_question" not in result.to_payload(debug=False, top_k=8)
