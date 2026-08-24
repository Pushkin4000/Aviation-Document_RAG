import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.engine import AviationRAGEngine
from app.retrieval import LexicalIndex

CORPUS = [
    ("QNH is the altimeter subscale setting which causes the altimeter to indicate "
     "elevation above mean sea level when the aircraft is on the ground.", 1),
    ("A cold front occurs when cold air replaces warm air at the surface, marked by "
     "a sharp change in wind direction and a fall in temperature.", 2),
    ("VOR stands for VHF Omni-directional Range, a navigation aid used to define "
     "airways and for en-route navigation.", 3),
    ("Catering arrangements for long haul cabin service on wide body aircraft.", 4),
]


@pytest.fixture
def settings():
    return Settings(_env_file=None, groq_api_key=None, generation_mode="extractive")


@pytest.fixture
def engine(settings):
    """An engine backed by an in-memory lexical index. No model, no network."""
    docs = [
        Document(page_content=text, metadata={"source": "fixture.pdf", "page": page,
                                              "chunk_id": f"fixture.pdf:p{page}:c1"})
        for text, page in CORPUS
    ]
    eng = AviationRAGEngine(settings=settings)
    eng.attach_index(vectorstore=None, lexical_index=LexicalIndex(docs))
    return eng
