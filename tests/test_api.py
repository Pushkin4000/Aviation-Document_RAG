import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.engine import AviationRAGEngine
from app.models import REFUSAL_MESSAGE
from app.retrieval import LexicalIndex


@pytest.fixture
def client(monkeypatch):
    import app.server as server

    docs = [
        Document(
            page_content="QNH is the altimeter subscale setting which causes the altimeter "
                         "to indicate elevation above mean sea level on the ground.",
            metadata={"source": "fixture.pdf", "page": 1, "chunk_id": "fixture.pdf:p1:c1"},
        )
    ]
    eng = AviationRAGEngine(settings=Settings(_env_file=None, groq_api_key=None, generation_mode="extractive"))
    eng.attach_index(vectorstore=None, lexical_index=LexicalIndex(docs))
    monkeypatch.setattr(server, "get_engine", lambda: eng)
    with TestClient(server.app) as c:
        yield c


def test_health_reports_index_and_generation_path(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["refusal_message"] == REFUSAL_MESSAGE
    assert body["index"]["lexical_entries"] == 1
    assert body["index"]["vector_loaded"] is False
    assert body["generation_path"] == "extractive"


def test_health_is_degraded_without_vectors(client):
    assert client.get("/health").json()["status"] == "degraded"


def test_health_reports_confidence_thresholds(client):
    settings = get_settings()
    body = client.get("/health").json()
    assert body["thresholds"] == {
        "clarify": settings.confidence_clarify_threshold,
        "answer": settings.confidence_answer_threshold,
    }


def test_ask_returns_answer_and_citations(client):
    body = client.post("/ask", json={"question": "What is QNH?"}).json()
    assert body["answer"]
    assert "route" in body and "confidence" in body and "decision" in body


def test_ask_refuses_out_of_scope(client):
    body = client.post("/ask", json={"question": "What is the best chocolate cake recipe?"}).json()
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["citations"] == []


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_rejects_overlong_question(client):
    assert client.post("/ask", json={"question": "x" * 5000}).status_code == 422


def test_debug_flag_returns_chunks(client):
    body = client.post("/ask", json={"question": "What is QNH?", "debug": True}).json()
    assert body["retrieved_chunks"]


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
