# Aviation Document RAG

Document-grounded RAG chatbot for AIRMAN aviation PDFs (PPL/CPL/ATPL/SOP/manual content), with strict refusal behavior and source citations.

## Assignment Coverage

### Level 1 Mandatory
- PDF ingestion pipeline with chunking + embeddings + FAISS persistence (`app/ingest.py`)
- API endpoints: `POST /ingest`, `POST /ask`, `GET /health` (`app/server.py`)
- Grounded answering with citations and strict refusal string (`app/graph.py`)
- 50-question evaluation set (`evaluation_set.json`)
- Evaluation script + report output (`evaluate.py` -> `evaluation_detailed.csv`, `report.md`)

### Level 2 Option 2 (Implemented)
- Query Router + Confidence Thresholding in `app/graph.py`
- Route `simple` questions to cheaper path and `complex` to stronger path
- Low-confidence branch triggers clarification prompt (plus mandatory refusal answer) or refusal

### Hard Refusal Rule
If evidence is insufficient, the assistant returns exactly:

`This information is not available in the provided document(s).`

## Tech Stack
- Python 3.10+
- FastAPI
- FAISS
- sentence-transformers (`all-MiniLM-L6-v2`)
- Optional generation model: Groq (`RAG_GENERATION_MODE=groq`)

## Project Layout
- `app/ingest.py`: load PDFs, clean text, chunk pages, build FAISS index
- `app/graph.py`: retrieval + grounding checks + answer/refusal logic
- `app/server.py`: FastAPI endpoints
- `evaluate.py`: batch evaluation and report generation
- `evaluation_set.json`: 50-question set (20 factual, 20 applied, 10 reasoning)

## Chunking Strategy
- Loader: `PyPDFLoader` (page-aware extraction)
- Cleaning: de-hyphenation across line breaks + whitespace normalization
- Splitter: `RecursiveCharacterTextSplitter`
  - `chunk_size = 900`
  - `chunk_overlap = 150`
- Metadata on each chunk:
  - `source` (document filename)
  - `page` (1-based)
  - `chunk_id` (`<file>:p<page>:c<index>`)

Rationale: 900/150 keeps enough context for aviation procedures while preserving retrieval precision and citation traceability.

## Incremental Vector DB (No Reupload)
- Ingestion keeps existing FAISS index and appends only new chunks.
- Chunk identity is computed with a stable fingerprint (`source + page + normalized text`).
- Registry file: `vectorstore/chunk_registry.json`.
- Default `/ingest` mode is incremental.
- Full rebuild is optional with `POST /ingest?rebuild=true`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
# Optional; required only when RAG_GENERATION_MODE=groq
GROQ_API_KEY=your_key_here
RAG_GENERATION_MODE=extractive
```

## Run

### 1) Ingest documents
Place AIRMAN PDFs in `data/`, then:

```bash
python app/ingest.py
```

or:

```bash
uvicorn app.server:app --reload
# POST /ingest           # incremental (default)
# POST /ingest?rebuild=true
```

### 2) Start API
```bash
uvicorn app.server:app --reload
```

### 3) Ask questions
`POST /ask`

```json
{
  "question": "What is QNH?",
  "debug": true
}
```

Response includes:
- `answer`
- `citations` (document + page)
- `route`, `confidence`, `decision`, optional `follow_up_question`
- `retrieved_chunks` (top chunks when `debug=true`)

## Option 2 Design Notes

### Confidence Calculation
Confidence is computed from top retrieved chunks:
- top relevance score
- average relevance score (top-3)
- top lexical overlap
- average lexical overlap (top-3)
- question-token coverage in top context
- noise penalty (MCQ/question-bank style text)

Final value is clamped to `[0,1]`.

### Routing Decisions
- Router output: `simple` or `complex`.
- Default router: heuristic (no extra API calls).
- Optional LLM router can be enabled with `RAG_ROUTER_MODE=groq` and `RAG_ROUTER_LLM_ENABLED=1`.
- Quota-safe default: `RAG_MODEL_ROUTING_ENABLED=0` keeps routing metadata active without extra Groq calls.
- If `RAG_MODEL_ROUTING_ENABLED=1`:
  - `simple` uses `RAG_SIMPLE_MODEL` (cheaper/smaller)
  - `complex` uses `RAG_COMPLEX_MODEL` (stronger)
- If model routing is disabled, both routes use local extractive grounding logic.

### Refusal / Clarification Trigger
- `confidence < RAG_CONFIDENCE_CLARIFY_THRESHOLD`: hard refusal.
- `RAG_CONFIDENCE_CLARIFY_THRESHOLD <= confidence < RAG_CONFIDENCE_ANSWER_THRESHOLD`: clarification branch by default.
- Mid-band salvage: if extractive answer is grounded and support strength exceeds `RAG_LOW_CONFIDENCE_SUPPORT_THRESHOLD`, answer is returned instead of refusing.
- `confidence >= RAG_CONFIDENCE_ANSWER_THRESHOLD`: answer path.

Grounding check uses both fuzzy similarity (`RAG_MIN_GROUNDED_SIMILARITY`) and answer-token overlap (`RAG_MIN_GROUNDED_TOKEN_OVERLAP`) against cited chunks.

Mandatory refusal string remains exactly:
`This information is not available in the provided document(s).`

## Evaluation

Run:

```bash
python evaluate.py
```

Outputs:
- `evaluation_detailed.csv`
- `report.md`

Metrics in report:
- Retrieval hit-rate
- Faithfulness
- Hallucination rate
- 5 best answers (with explanation)
- 5 worst answers (with explanation)
