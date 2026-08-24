from langchain_core.documents import Document

from app.config import Settings
from app.models import RetrievedChunk
from app.retrieval import LexicalIndex, Retriever

SETTINGS = Settings(_env_file=None)


def _doc(text, page=1, cid="s.pdf:p1:c1"):
    return Document(page_content=text, metadata={"source": "s.pdf", "page": page, "chunk_id": cid})


def _index():
    return LexicalIndex(
        [
            _doc("QNH is the altimeter subscale setting to obtain elevation above mean sea level.", 1, "s.pdf:p1:c1"),
            _doc("A cold front occurs when cold air replaces warm air at the surface.", 2, "s.pdf:p2:c1"),
            _doc("Catering arrangements for long haul cabin service.", 3, "s.pdf:p3:c1"),
        ]
    )


def test_lexical_search_ranks_relevant_first():
    hits = _index().search("What is QNH?", top_k=3, settings=SETTINGS)
    assert hits
    assert "QNH" in hits[0].document.page_content


def test_lexical_search_returns_nothing_for_unrelated_query():
    assert _index().search("chocolate cake recipe", top_k=3, settings=SETTINGS) == []


def test_retriever_uses_lexical_when_vectorstore_absent():
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=SETTINGS)
    assert r.retrieve("What is QNH?")


def test_select_candidates_returns_empty_for_irrelevant_chunks():
    """Regression for bug #2.

    The original admitted any chunk at idx < 2 regardless of score or
    overlap, so candidates were never empty and the no-evidence refusal
    branch was unreachable.
    """
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=SETTINGS)
    junk = [
        RetrievedChunk(document=_doc("Catering arrangements for cabin service."), score=0.01, lexical_overlap=0.0),
        RetrievedChunk(document=_doc("Baggage handling procedures at the ramp."), score=0.01, lexical_overlap=0.0),
    ]
    candidates, tokens = r.select_candidates("What is the boiling point of water?", junk)
    assert candidates == []
    assert tokens


def test_select_candidates_keeps_relevant_chunks():
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=SETTINGS)
    good = [RetrievedChunk(document=_doc("QNH is the altimeter subscale setting."), score=0.9)]
    candidates, _ = r.select_candidates("What is QNH?", good)
    assert len(candidates) == 1


def test_select_candidates_respects_max_candidates():
    settings = Settings(_env_file=None, max_candidate_chunks=2)
    r = Retriever(vectorstore=None, lexical_index=_index(), settings=settings)
    many = [RetrievedChunk(document=_doc("QNH altimeter setting mean sea level."), score=0.9) for _ in range(6)]
    candidates, _ = r.select_candidates("What is QNH?", many)
    assert len(candidates) == 2
