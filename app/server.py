from dataclasses import asdict
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.graph import REFUSAL_MESSAGE, rag_engine
from app.ingest import IngestSummary, ingest_pipeline

app = FastAPI(title="AIRMAN Aviation Document RAG")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    debug: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    page: int
    score: float
    lexical_overlap: Optional[float] = None
    content_snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    route: Optional[str] = None
    confidence: Optional[float] = None
    decision: Optional[str] = None
    follow_up_question: Optional[str] = None
    retrieved_chunks: Optional[List[RetrievedChunk]] = None


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "index_loaded": (rag_engine.vectorstore is not None) or bool(rag_engine.lexical_entries),
        "refusal_message": REFUSAL_MESSAGE,
    }


@app.post("/ingest")
def run_ingest(rebuild: bool = Query(default=False)) -> dict:
    try:
        summary: IngestSummary = ingest_pipeline(rebuild=rebuild)
        rag_engine.refresh_index()
        return {
            "status": "completed",
            "mode": "rebuild" if rebuild else "incremental",
            "summary": asdict(summary),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask_question(request: AskRequest) -> AskResponse:
    result = rag_engine.ask(request.question, debug=request.debug)
    return AskResponse(
        answer=str(result.get("answer", REFUSAL_MESSAGE)),
        citations=list(result.get("citations", [])),
        route=result.get("route"),
        confidence=result.get("confidence"),
        decision=result.get("decision"),
        follow_up_question=result.get("follow_up_question"),
        retrieved_chunks=result.get("retrieved_chunks"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
