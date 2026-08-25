from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.engine import get_engine
from app.ingest import IngestSummary, ingest_pipeline
from app.logging_setup import configure_logging, get_logger
from app.models import REFUSAL_MESSAGE

logger = get_logger("server")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = get_engine()  # loads the index once, at startup rather than at import
    state = engine.index_state
    if state.degraded_reason:
        logger.warning("Started DEGRADED: %s", state.degraded_reason)
    else:
        logger.info("Started: %d lexical entries, generation via %s",
                    state.lexical_entries, engine.generation_path)
    yield


app = FastAPI(title="AIRMAN Aviation Document RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    debug: bool = False


class RetrievedChunkPayload(BaseModel):
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
    # Which generator actually answered THIS request: "groq", "extractive",
    # or "n/a" when the query was refused before generation. /health reports
    # the configured path, which can differ per request -- a rejected key or
    # an exhausted daily token cap makes a "groq" service answer
    # extractively. Undeclared, FastAPI's response_model silently strips
    # this, leaving callers unable to tell the two apart.
    generation_path: Optional[str] = None
    follow_up_question: Optional[str] = None
    retrieved_chunks: Optional[List[RetrievedChunkPayload]] = None


@app.get("/health")
def health_check() -> dict:
    settings = get_settings()
    engine = get_engine()
    state = engine.index_state
    healthy = state.vector_loaded and state.degraded_reason is None
    return {
        "status": "ok" if healthy else "degraded",
        "index": asdict(state),
        "generation_path": engine.generation_path,
        "refusal_message": REFUSAL_MESSAGE,
        "thresholds": {
            "clarify": settings.confidence_clarify_threshold,
            "answer": settings.confidence_answer_threshold,
        },
    }


@app.post("/ingest")
def run_ingest(rebuild: bool = Query(default=False)) -> dict:
    try:
        summary: IngestSummary = ingest_pipeline(rebuild=rebuild)
    except ValueError as exc:
        logger.warning("Ingest rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    get_engine().refresh_index()
    return {
        "status": "completed",
        "mode": "rebuild" if rebuild else "incremental",
        "summary": asdict(summary),
    }


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = get_engine().ask(request.question, debug=request.debug)
    except Exception as exc:
        logger.exception("Query failed for %r", request.question[:80])
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return AskResponse(**result)


@app.get("/")
def index() -> FileResponse:
    """Animated landing page. The working instrument lives at /console."""
    return FileResponse(WEB_DIR / "landing.html")


@app.get("/console")
def console() -> FileResponse:
    """The Night Cockpit query console — the page that actually queries /ask."""
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
