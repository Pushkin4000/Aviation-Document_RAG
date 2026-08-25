# AIRMAN — Aviation Document RAG

## 1. What this is

A document-grounded RAG service that answers questions strictly from a corpus
of aviation training PDFs (PPL/CPL/ATPL theory, SOPs, meteorology, flight
planning) and cites the source document and page for every claim. Retrieval
combines a FAISS vector index with a lexical fallback; generation is either
Groq-hosted (`llama-3.3-70b-versatile`, JSON-constrained, context-only) or a
local extractive path that returns a verbatim, cleaned span of the cited
chunk. When the retrieved evidence does not support an answer, the service
refuses rather than guesses, returning exactly:

> `This information is not available in the provided document(s).`

## 2. Quick start

**POSIX (bash/zsh):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit GROQ_API_KEY / RAG_GENERATION_MODE
uvicorn app.server:app --reload
```

**Windows (PowerShell / cmd):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.server:app --reload
```

The repo ships a pre-built `vectorstore/` (16 MB), so the service is queryable
immediately after `pip install`. To re-ingest from source PDFs, place them in
`data/` (not tracked in git — see Known Limitations) and call `POST
/ingest`.

## 3. Configuration

Every setting is a `RAG_*` (or `GROQ_API_KEY` / `HF_LOCAL_FILES_ONLY`)
environment variable, read once at startup via `app/config.py`. Defaults
below match the code, not aspiration.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_DATA_DIR` | `data` | Source PDF directory for ingestion. |
| `RAG_VECTORSTORE_DIR` | `vectorstore` | FAISS index + registry location. |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name. |
| `RAG_EMBEDDING_DIMENSION` | `384` | Expected vector width; mismatches raise loudly instead of silently degrading. |
| `HF_LOCAL_FILES_ONLY` | `0` (`False`) | **Forces offline model loading — see warning below.** |
| `RAG_TOP_K` | `8` | Chunks returned by the retriever. |
| `RAG_MIN_RELEVANCE` | `0.35` | Minimum vector similarity to keep a chunk as a candidate. |
| `RAG_MIN_CHUNK_LEXICAL_OVERLAP` | `0.12` | Minimum token overlap to keep a chunk as a candidate. |
| `RAG_MIN_SEGMENT_SCORE` | `0.23` | Minimum score for a text segment inside a chunk to be usable. |
| `RAG_MAX_CANDIDATE_CHUNKS` | `5` | Cap on chunks passed into generation. |
| `RAG_MIN_GROUNDED_SIMILARITY` | `0.65` | Fuzzy-match floor for judging an answer grounded in its citation. |
| `RAG_MIN_GROUNDED_TOKEN_OVERLAP` | `0.58` | Token-overlap floor for the same grounding check. |
| `RAG_ANSWER_MAX_WORDS` | `65` | Hard cap on extractive answer length. |
| `RAG_CONFIDENCE_ANSWER_THRESHOLD` | `0.48` | Confidence at/above which the system answers outright. |
| `RAG_CONFIDENCE_CLARIFY_THRESHOLD` | `0.34` | Confidence below which the system refuses outright; between this and the answer threshold it clarifies or salvages (see below). |
| `RAG_LOW_CONFIDENCE_SUPPORT_THRESHOLD` | `0.66` | In the clarify band, an answer is returned anyway if its grounding support exceeds this. |
| `RAG_GENERATION_MODE` | `groq` | `groq` or `extractive`. `generation_path` in `/health` reports `extractive` whenever no `GROQ_API_KEY` is present, regardless of this setting — but that check only tests *presence*, not validity: a present-but-invalid key (as in this environment) still reports `groq` in `/health` and only falls back to extractive per-request, after each Groq call fails. See Known Limitations. |
| `GROQ_API_KEY` | *(unset)* | Groq API key. Required for the `groq` path to actually run. |
| `RAG_GROQ_MODEL` | `llama-3.3-70b-versatile` | Model used for generation. |
| `RAG_GROQ_TIMEOUT_SECONDS` | `30.0` | Groq request timeout. |
| `RAG_ROUTER_MODE` | `heuristic` | `heuristic` or `groq` (LLM-assisted routing). |
| `RAG_ROUTER_LLM_ENABLED` | `0` (`False`) | Gate for the LLM router. |
| `RAG_MODEL_ROUTING_ENABLED` | `0` (`False`) | If set, `simple`/`complex` routes use different Groq models. |
| `RAG_SIMPLE_MODEL` | `llama-3.1-8b-instant` | Model for the `simple` route when model routing is enabled. |
| `RAG_COMPLEX_MODEL` | `llama-3.3-70b-versatile` | Model for the `complex` route when model routing is enabled. |
| `RAG_JUDGE_MODEL` | `llama-3.3-70b-versatile` | Model used by `evaluate.py` to grade correctness/faithfulness. |
| `RAG_CHUNK_SIZE` | `900` | Characters per chunk at ingestion. |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks. |
| `RAG_FILTER_EXERCISE_CHUNKS` | `1` (`True`) | Drop question-bank/MCQ-style chunks during ingestion. |
| `RAG_LOG_LEVEL` | `INFO` | Logging verbosity. |

**`HF_LOCAL_FILES_ONLY` warning.** Setting this to `1` forbids
`sentence-transformers` from reaching huggingface.co, so it only works if
`all-MiniLM-L6-v2` is *already* present in the local HF cache
(`$HF_HOME`/`~/.cache/huggingface`). This is not hypothetical: in the
version of this service that predates this branch, the model was **not**
cached, `HF_LOCAL_FILES_ONLY=1` was set anyway, `SentenceTransformerEmbeddings`
raised `OSError` on load, and a bare `except Exception` in the old code
swallowed it — the service kept running, silently, on keyword-only lexical
matching for the entire life of every process, with no error, no log, and no
signal in `/health`. That failure mode is why `app/embeddings.py` now raises
a distinct `EmbeddingsUnavailable` exception instead of catching broadly, and
why `/health` reports `vector_loaded` and a `degraded_reason` explicitly. The
Docker image sidesteps the whole problem by baking the model in at build
time, *then* setting `HF_LOCAL_FILES_ONLY=1` (see Deployment) — never set it
in an environment where the model has not been downloaded first.

## 4. Architecture

`app/graph.py`, formerly a 1,108-line monolith, is now a 22-line
backwards-compatible re-export shim. The real implementation is split by
responsibility:

| Module | Responsibility |
|---|---|
| `app/config.py` | Single `Settings` object (pydantic-settings) — every `RAG_*` env var, validated once at startup. |
| `app/models.py` | Shared types: `Decision`, `GenerationOutcome`, `RetrievedChunk`, `GenerationResult`, `AskResult`, and the `REFUSAL_MESSAGE` constant. |
| `app/embeddings.py` | Builds the `SentenceTransformerEmbeddings` wrapper; raises `EmbeddingsUnavailable` loudly instead of degrading silently; verifies vector dimension against `RAG_EMBEDDING_DIMENSION`. |
| `app/ingest.py` | PDF loading (page-aware), text cleaning, chunking, incremental FAISS index build via `chunk_registry.json`. |
| `app/retrieval.py` | Hybrid retrieval: `Retriever` (vector + lexical candidate selection) and `LexicalIndex` (keyword fallback index). |
| `app/scoring.py` | Tokenization, lexical overlap, noise penalty, definition-question heuristics used by both retrieval and generation. |
| `app/routing.py` | Heuristic (and optional LLM) `simple`/`complex` question routing. |
| `app/generation/extractive.py` | Local, network-free answer extraction: definition probes, candidate ranking, answer cleaning/length capping. |
| `app/generation/groq_generator.py` | Groq-backed generation: JSON-constrained prompt, context built only from cited chunks, isolated `_invoke_chain` for test substitution. |
| `app/grounding.py` | Post-hoc check that a generated answer is actually supported by its cited chunk (fuzzy similarity + token overlap). |
| `app/engine.py` | `AviationRAGEngine` — orchestrates retrieve → route → generate → ground → decide; lazy index load at first use (or explicit startup), not at import time. |
| `app/judge.py` | LLM-as-judge (with lexical fallback and on-disk cache) used only by `evaluate.py`, never in the request path. |
| `app/server.py` | FastAPI app: `/health`, `/ask`, `/ingest`, `/`, static frontend mount. |
| `app/graph.py` | Re-export shim kept for import-path compatibility (`from app.graph import ...`). |
| `app/logging_setup.py` | Idempotent logging configuration. |

**Why this mattered for import cost.** The previous `app/graph.py` built its
FAISS index and embeddings model as a module-level singleton, so simply
`import app.graph` — as every test file did — paid the full model-load cost.
Moving index construction into `AviationRAGEngine`, loaded lazily via
`get_engine()` at first use (or explicitly in the FastAPI `lifespan` startup
hook) rather than at import time, cut the cost of importing the app from
**47.9s to ~1.5s**.

## 5. API

### `GET /health`

```
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "index": {
    "vector_loaded": true,
    "lexical_entries": 6590,
    "degraded_reason": null
  },
  "generation_path": "extractive",
  "refusal_message": "This information is not available in the provided document(s).",
  "thresholds": {
    "clarify": 0.34,
    "answer": 0.48
  }
}
```

`status` is `"degraded"` whenever `vector_loaded` is false or
`degraded_reason` is set — e.g. a missing/uncached embedding model, a
missing `vectorstore/` directory, or a FAISS load error. `generation_path`
reflects the *effective* mode (`groq` only when `RAG_GENERATION_MODE=groq`
**and** a `GROQ_API_KEY` is present), not just the configured setting.

### `POST /ask`

```
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is QNH in altimetry?", "debug": false}'
```

```json
{
  "answer": "QNH is always rounded down to the nearest integer.",
  "citations": ["Meteorology full book.pdf (Page 135)"],
  "route": "simple",
  "confidence": 1.0,
  "decision": "answer"
}
```

Set `"debug": true` to also receive `retrieved_chunks` (chunk id, source,
page, score, lexical overlap, content snippet) for the top-k candidates. A
refused question returns `decision` starting with `refuse_` (there are
several distinct refusal reasons — empty question, no relevant chunks, low
confidence, no supported answer, grounding failed, model declined) and
`answer` equal to the exact refusal string above; a mid-confidence question
may instead come back as `decision: "clarify_low_confidence"` with a
`follow_up_question`.

### `POST /ingest`

```
curl -X POST "http://127.0.0.1:8000/ingest?rebuild=false"
```

```json
{
  "status": "completed",
  "mode": "incremental",
  "summary": {
    "files_indexed": 7,
    "pages_loaded": 3526,
    "chunks_total_after_filter": 4959,
    "chunks_added": 0,
    "chunks_skipped_existing": 4959,
    "chunks_dropped": 1631,
    "chunk_size": 900,
    "chunk_overlap": 150,
    "embedding_model": "all-MiniLM-L6-v2",
    "vectorstore_dir": "vectorstore"
  }
}
```

Default mode is incremental (identifies existing chunks by a
`source + page + normalized text` fingerprint and only appends new ones);
`?rebuild=true` rebuilds the index from scratch. Requires PDFs in
`RAG_DATA_DIR` (`data/` by default) — not present in a container built from
this repo unless supplied separately (see Known Limitations). `chunks_dropped`
counts exercise/MCQ-style chunks removed by `RAG_FILTER_EXERCISE_CHUNKS`; the
committed `vectorstore/` (6,590 entries) was built with that filter disabled,
which is why `chunks_total_after_filter` above (4,959) is lower than the
index's actual `lexical_entries` count reported by `/health`.

## 6. Evaluation

`evaluate.py` runs all 50 questions in `evaluation_set.json` plus all 15 in
`out_of_scope_set.json` against a live `AviationRAGEngine`, and writes
`evaluation_detailed.csv` and `report.md`.

**Ground truth.** All 50 in-scope questions were authored (question, expected
answer, expected source document/page, key facts) against the actual indexed
corpus by retrieving and reading the relevant pages, not generated
speculatively. 9 of the 50 are deliberately marked `unanswerable: true`
because the corpus genuinely does not contain an answer — the system is
expected to refuse those. `out_of_scope_set.json` is a separate 15-question
set (e.g. "What is the best recipe for chocolate cake?") that is entirely
off-topic and must always be refused; it is kept separate so the original
50-question submission set stays intact.

**What each metric means:**
- **Retrieval recall@k** — did the retriever surface a chunk from a page the
  ground truth says actually contains the answer?
- **Answer correctness** — does the generated answer convey the reference's
  key facts, per an LLM judge (falls back to lexical matching when no judge
  model is reachable)?
- **Faithfulness / hallucination rate** — is every claim in the answer
  supported by the chunk(s) it *cited* (not everything retrieved)?
- **Refusal recall / precision** — of the 15 out-of-scope questions, how many
  were correctly refused (recall); of all refusals issued, how many were on
  questions that should have been refused (precision).
- **Latency p50/p95** — wall-clock per `/ask` call during the eval run.

**Current honest numbers** (`RAG_GENERATION_MODE=extractive`, from `report.md`):

| Metric | Value |
|---|---|
| Retrieval recall@k | 68.0% |
| Answer correctness | 20.0% |
| Faithfulness (of answered) | 100.0% — **not meaningful**, see below |
| Hallucination rate (of answered) | 0.0% — **not meaningful**, see below |
| Refusal recall | 13/15 (86.7%) |
| Refusal precision | 55.6% |
| Latency p50 / p95 | ~180 ms / ~390 ms |

**On the 100%/0% faithfulness figures — read this before trusting them.** An
earlier version of this evaluation compared each answer against the same
chunk text it had been extracted from, which cannot fail: it always reported
100% faithfulness and 0% hallucination, regardless of whether the system
worked. **Those old figures were an artifact of a circular metric, not a
property of the system, and are not comparable to anything in this
README.** The rebuilt evaluation still shows 100%/0% under
`RAG_GENERATION_MODE=extractive`, but for a legitimate structural reason
this time: an extractive answer is, by construction, a verbatim span copied
out of its own cited chunk, so checking it against that chunk cannot fail
either. Faithfulness and hallucination only become informative under
`RAG_GENERATION_MODE=groq`, where the model is free to introduce claims the
context doesn't contain. The metrics that do carry real signal in the
current, extractive-mode run are **retrieval recall@k**, **answer
correctness**, and the **refusal** figures — and a correctness score of
20% against a recall of 68% shows the honest gap plainly: retrieval mostly
finds the right page, but extractive generation frequently fails to produce
the right answer from it.

## 7. Frontend

The UI ("Night Cockpit") takes its palette and layout from cockpit
instrument flood lighting — aviation panels are lit dim, warm-amber at night
specifically to preserve a pilot's dark adaptation, since white light in a
dark cockpit destroys night vision. That functional constraint, not a
preference for dark mode, is why the interface commits to a dark ground with
a single warm accent and offers no light theme.

| Role | Token | Hex |
|---|---|---|
| Ground | `--ground` | `#0F0B08` |
| Panel | `--panel` | `#1B1612` |
| Raised | `--raised` | `#25211B` |
| Hairline | `--hairline` | `#37322C` |
| Rule | `--rule` | `#4C4741` |
| Text primary | `--text` | `#F0ECE7` |
| Text secondary | `--text-2` | `#BBB6B0` |
| Text muted | `--muted` | `#8A857F` |
| Text faint | `--faint` | `#625D57` |
| Accent 100 | `--accent-100` | `#51321E` |
| Accent 200 | `--accent-200` | `#854E20` |
| Accent 300 | `--accent-300` | `#B56C15` |
| Accent 400 (live state / primary action) | `--accent-400` | `#E29019` |
| Accent 500 | `--accent-500` | `#FAB550` |
| Accent 600 | `--accent-600` | `#FFD795` |
| Refusal | `--refusal` | `#B6604E` |
| Grounded | `--grounded` | `#99A668` |
| Caution | `--caution` | `#D7A03D` |

Served as static files mounted directly by FastAPI (`app/server.py`) — no
bundler, no build step. `GET /` returns `web/index.html`; `web/app.js` and
`web/styles.css` are mounted at `/static`.

## 8. Deployment

```bash
docker build -t airman-rag .
docker run --rm -p 8000:8000 -e GROQ_API_KEY="$GROQ_API_KEY" airman-rag
```

The embedding model is baked into the image at build time (`RUN python -c
"...SentenceTransformer('all-MiniLM-L6-v2')"`), and only *after* that does
the Dockerfile set `HF_LOCAL_FILES_ONLY=1`. That ordering is deliberate and
must not be reversed: it is what makes `HF_LOCAL_FILES_ONLY=1` safe inside
the container (see the warning in Configuration) — the model is already on
disk before the flag ever takes effect, so the running container never needs
to reach huggingface.co. The image copies in the committed `vectorstore/`
(16 MB) rather than the 527 MB `data/` source PDFs, ships without them
(`.dockerignore` excludes `data/`), and runs as a non-root user (uid 10001).
A `HEALTHCHECK` polls `GET /health` and expects `status: "ok"`.

## 9. Testing

```bash
pip install -r requirements-dev.txt
pytest
```

85 tests, all offline — none requires network access or a Groq API key
(Groq calls are isolated behind `_invoke_chain` and substituted in tests).

## 10. Known limitations

- **No light theme.** The frontend is a deliberately single-themed,
  night-adapted instrument (see Frontend); it is not legible in bright
  ambient light and that is an accepted trade-off, not a bug.
- **Ground truth was authored via retrieval.** The 50-question evaluation
  set's expected answers and source pages were written by reading the
  corpus pages the retriever surfaced, not by an independent SME pass —
  treat correctness figures as internally consistent, not externally
  audited.
- **`data/` is untracked.** The 527 MB source PDF corpus is not in git; a
  container or checkout built from this repo can serve queries against the
  committed `vectorstore/` but cannot re-ingest until `data/` is supplied
  separately and `POST /ingest` is called.
- **The configured `GROQ_API_KEY` is invalid** (the key on file returns HTTP
  401), so `RAG_GENERATION_MODE` is effectively `extractive` in this
  environment and the Groq generation path — while implemented and covered
  by unit tests with the network call substituted — has never actually run
  live end-to-end. All metrics in Section 6 were produced under extractive
  generation. **To exercise the Groq path:** supply a valid `GROQ_API_KEY`
  and set `RAG_GENERATION_MODE=groq`; `GET /health`'s `generation_path`
  field will then read `"groq"` once both are true.
