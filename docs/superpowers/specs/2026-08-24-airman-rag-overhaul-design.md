# AIRMAN Aviation RAG — Overhaul Design

Date: 2026-08-24
Status: approved (pending spec review)
Repo: https://github.com/Pushkin4000/AIRMAN---assingment

## Goal

Take a submitted assignment prototype and make it fast, reliable, honestly
measured, and deployable, with a frontend that exposes the retrieval pipeline
rather than hiding it.

The hard product rule does not change. When evidence is insufficient the API
returns exactly:

    This information is not available in the provided document(s).

## Why this work is needed

Findings from the audit of commit `165aafb`.

### The evaluation is circular

`evaluate.py:66` scores faithfulness by fuzzy-matching the answer against the
retrieved chunks. `_answer_extractive` builds the answer by copying spans out of
those same chunks. The metric therefore compares text against its own source and
cannot fail. The reported "Faithfulness 100%, Hallucination 0%" measures nothing.

`evaluation_set.json` has no ground truth, so "retrieval hit-rate" is only token
overlap between the question and the retrieved text. A retriever that returned
the question back verbatim would score 100%.

The consequence is visible in `report.md`, which lists this among the five *best*
answers:

    temperature of 2°C and a dew point of 2°C there must be uniform fog d.

That is a fragment of a multiple-choice exam question, ranked as a top result.

### Correctness bugs

1. **LLM refusal is silently overridden.** `_answer_with_groq_model` returns
   `(REFUSAL_MESSAGE, [])` when the model correctly declines. The caller at
   `graph.py:475` tests `if llm_answer and llm_used:` — the empty list is falsy,
   so control falls through to `_answer_extractive`, which may answer anyway.
   The model's refusal is discarded.
2. **The no-evidence refusal path is dead code.** `_select_candidate_chunks`
   admits any chunk with `idx < 2` regardless of score or overlap, so
   `candidate_chunks` is never empty. `refuse_no_relevant_chunks` is unreachable.
3. **Failures are invisible.** Roughly fifteen `except Exception: pass` blocks
   and no logging anywhere. A corrupt FAISS index sets `self.vectorstore = None`
   and the service then answers from lexical search alone, indefinitely, with no
   error surfaced through `/health` or logs.

### Structural problems

- `app/graph.py` is 1,108 lines in a single class spanning retrieval, routing,
  confidence, extraction, grounding, and response assembly.
- `rag_engine = AviationRAGEngine()` executes at import time, loading FAISS and
  sentence-transformers as a side effect of importing the module. This makes
  import slow, unit testing impractical, and process startup opaque.
- Twenty configuration values are read via `os.getenv` into module-level globals
  at import. Nothing can be overridden per-call or per-test.
- `evaluate.py` reaches into `rag_engine._retrieve(...)`, a private method, and
  re-runs retrieval a second time for every question, doubling evaluation cost.
- `STOP_WORDS` and `tokenize` are duplicated between `graph.py` and
  `evaluate.py`, with different contents, so the evaluator tokenizes differently
  than the system under test.
- `SentenceTransformerEmbeddings` from `langchain_community` is deprecated.
- Both `graph.py` and `ingest.py` manually `pickle.load` FAISS's `index.pkl`
  instead of reading `vectorstore.docstore._dict` after a normal load.

### Repository hygiene

- No `.gitignore`. `.env` holds a live `GROQ_API_KEY` and is one `git add .`
  away from being published. (Verified: no key is in history today. The
  committed `.env.example` had an empty value and `gsk_` appears in no revision.)
- 40+ `.pyc` files are tracked, including 31 copies of
  `graph.cpython-313.pyc.<pointer>` — Windows import-lock debris.
- `.env.example` is deleted in the working tree while `README.md` still
  instructs the reader to copy it.
- `data/Meteorology full book.pdf` (175 MB) is untracked, yet the committed
  vectorstore and `report.md` both depend on it. Results are unreproducible.
- `requirements.txt` pins no versions.
- `.venv/` and `venv/` both exist in the working tree.
- `data/` totals 527 MB and six PDFs are already in history.

## Scope

In scope: backend restructure, Groq as the primary generation path, honest
evaluation with ground truth, a frontend, tests, packaging for deploy, and
repository hygiene.

Out of scope, deliberately:

- **Rewriting git history to purge the 527 MB of PDFs.** The repo is already
  submitted and pushed; a force-push would invalidate the graded reference.
  Going forward `data/` is ignored. The committed vectorstore (16 MB) lets the
  service run without the PDFs present, so deploys are unaffected. This can be
  revisited as an isolated task.
- Replacing FAISS, changing the embedding model, or re-chunking the corpus. The
  existing index stays valid; ingestion remains incremental.
- Authentication, rate limiting, and multi-user state.

## Architecture

### Module layout

    app/
      config.py       Settings (pydantic-settings). Single source of truth.
      logging_setup.py  Structured logging setup.
      models.py       RetrievedChunk, AskResult, Decision enum.
      scoring.py      tokenize, overlap, noise_penalty, confidence, support.
      retrieval.py    Hybrid vector + lexical retrieval, candidate selection.
      routing.py      Query router (heuristic default, optional LLM).
      generation.py   Groq generation (primary) + extractive (fallback).
      grounding.py    Post-generation grounding verification.
      engine.py       Orchestration. Lazy get_engine().
      ingest.py       PDF -> chunks -> FAISS. Largely retained.
      server.py       FastAPI routes + static mount.
    web/
      index.html, app.js, styles.css
    tests/
      test_scoring.py test_grounding.py test_routing.py
      test_retrieval.py test_generation.py test_api.py test_ingest.py

Evaluation artifacts stay at the repository root — `evaluate.py`,
`evaluation_set.json`, `evaluation_detailed.csv`, `report.md` — because the
submitted README and the assignment brief reference those exact paths. The new
`out_of_scope_set.json` joins them there rather than introducing an `eval/`
package that would invalidate the submitted layout.

`app/graph.py` is retained as a thin re-export shim so existing imports and the
submitted README keep working.

### Three structural changes that carry the weight

**Lazy initialisation.** `get_engine()` replaces the module-level singleton,
wired to a FastAPI `lifespan` handler so the index loads once at startup rather
than at import. Tests construct an engine against a fixture index.

**Configuration as an object.** A `Settings` instance replaces the twenty
globals, injected into the engine. Tests override by constructing `Settings`
directly. `.env` continues to populate defaults.

**Errors become visible.** Every swallowed exception is logged with context.
Index-load failure sets an explicit degraded state that `/health` reports, so a
broken index is observable instead of silently downgrading to lexical-only.

### Request flow

    question
      -> retrieve      hybrid vector + lexical, top_k
      -> select        candidate chunks (relevance/overlap gate, bug #2 fixed)
      -> route         simple | complex
      -> confidence    scored from top-3 chunks
      -> gate          < clarify: refuse
                       < answer:  salvage-or-clarify
                       >=answer:  generate
      -> generate      Groq if key present, else extractive
      -> ground        verify against cited chunks; refuse on failure
      -> respond       answer + citations + telemetry

The grounding gate runs on the complete answer, after generation. This is why
the response is not streamed: streaming would display text to the user before it
has been verified as supported, which contradicts the product's central rule.
The frontend instead reports live pipeline stages.

### Generation

Groq becomes the primary path; extraction becomes the fallback when no API key
is configured. This is the change that fixes answer quality — extraction can
only ever quote fragments of the PDF.

Bug #1 is fixed by returning a typed result from the generator so that "the
model refused" is distinguishable from "the model failed", rather than both
collapsing to a falsy value.

Model routing keeps its current shape: `simple` -> `RAG_SIMPLE_MODEL`,
`complex` -> `RAG_COMPLEX_MODEL`, both configurable, disabled by default to stay
quota-safe.

## Evaluation

Ground truth is authored by **retrieve-then-verify**: for each of the 50
questions, pull top chunks from the existing index, write the expected answer
from what the corpus actually says, and record source and page.

Each entry gains:

    expected_answer   str        reference answer written from the corpus
    expected_sources  [{source, page}]
    key_facts         [str]      facts a correct answer must contain

Metrics:

| Metric | Method |
|---|---|
| Retrieval recall@k | expected source/page appears in top-k |
| Answer correctness | LLM judge, answer vs. expected_answer + key_facts |
| Faithfulness | LLM judge, every claim supported by the **cited** chunks |
| Refusal precision | of all refusals, how many were correct to refuse |
| Refusal recall | of out-of-scope questions, how many were refused |
| Latency p50/p95 | per request |

Faithfulness is judged against the chunks the answer **cited**, not against
everything retrieved. That distinction is what breaks the current circularity.

The judge is Groq, model `RAG_JUDGE_MODEL` (default
`llama-3.3-70b-versatile`), called at temperature 0 and required to return a
structured verdict with a reason, so every score in `evaluation_detailed.csv` is
auditable rather than an opaque number. Judge calls are cached by
question-plus-answer hash so re-runs do not re-spend quota.

Without an API key the evaluator degrades to embedding-similarity scoring
against `expected_answer` and exact matching on `key_facts`, and labels the
report accordingly. It never silently reports judge-quality numbers produced by
the fallback.

The out-of-scope set is new and necessary: the system's defining behaviour is
refusal, and today not one test case is supposed to be refused. It is kept as a
separate file so the original 50 remain intact as the submission record.

Reported numbers are expected to fall. The current ones are artifacts of the
measurement, not properties of the system. `report.md` will state the
methodology and note that figures are not comparable to the previous run.

## Frontend — "Night Cockpit"

**Referent.** Cockpit instrument flood lighting. Aviation panels are lit dim and
warm-amber at night to preserve the pilot's dark adaptation; white light in a
dark cockpit destroys night vision. The dark ground and single warm accent
therefore descend from a functional constraint in the product's own domain, not
from a preference for dark mode.

**The sacrifice.** The interface is illegible in bright sunlight and offers no
light theme. That is accepted: it is a night-adapted instrument, and committing
to that is what makes the palette mean something.

### Colour

Derived in OKLCH, converted to sRGB. Neutrals carry the accent hue at low
chroma; the accent shifts hue 52 deg -> 84 deg across its ramp with chroma
peaking mid-lightness and tapering at both ends, the way a real pigment behaves.

Surfaces (hue 66-72, chroma 0.010-0.013):

    ground    #0F0B08     panel     #1B1612     raised    #25211B
    hairline  #37322C     rule      #4C4741

Text (warm off-whites; contrast against ground):

    primary   #F0ECE7  16.66:1     secondary #BBB6B0  9.73:1
    muted     #8A857F   5.36:1     faint     #625D57  3.01:1

Accent, sodium vapour amber:

    100 #51321E   200 #854E20   300 #B56C15   400 #E29019
    500 #FAB550   600 #FFD795   700 #FFF0D0

Semantics, derived from the same family rather than imported from a framework:

    refusal  #B6604E  (brick)    grounded #99A668  (olive)
    caution  #D7A03D  (ochre)

Verified: no collision with the indigo/violet range, slate-900/950, or stock
Tailwind semantic values; no `R=G=B` neutral; no `#FFFFFF` or `#000000`.

**Accent scarcity.** `accent-400` has exactly one job: live state and the
primary action. Target under 5% of painted pixels. Everything else works through
weight, size, spacing, and neutral value.

### Typography

Two faces separated by classification:

- **Archivo** (variable grotesque, real width axis) carries the voice.
- **IBM Plex Mono** carries the apparatus — chunk IDs, page numbers, scores,
  confidence values, the decision enum. This is functional: the interface is
  mostly identifiers and numerics, which want tabular figures.

Weights 400 and 700 only. Tracking -0.03em on display sizes, +0.08em on small
mono labels. Line-height moves inversely with size, 1.1 at display to 1.6 at
body. Body measure capped at 68ch.

### Shape and surface

Radius 0 throughout. Instrument bezels and approach plates are square; sharp is
the committed structural position, not an oversight. No shadows — depth comes
from 1px hairlines in tinted neutral and tonal background steps. No
glassmorphism, no gradients, no decorative background geometry.

### Layout

A query console, not a landing page. Asymmetric two-column, flush left:

- **Left (2/3)** — the question, the answer set large and sparse, citations
  beneath as source + page.
- **Right (1/3)** — telemetry rail, dense and small, set in mono: route,
  confidence plotted against its threshold bands, decision enum, retrieved
  chunks with scores, and which of them were actually cited.

Density is deliberately unequal between the columns; that contrast is what
creates emphasis without adding colour.

The refusal string gets a distinct treatment in brick — it is the product's
thesis, not an error state.

### Motion

Motion exists only to report request state: the pipeline stage indicator
advancing through retrieving -> routing -> generating -> verifying.

**Nothing else animates.** No scroll-triggered reveals, no entrance
transitions, no hover elevation. Durations: 120ms for state changes, 200ms for
the stage indicator.

### Serving

Static files mounted directly by FastAPI. No Vite, no bundler, no build step, no
CORS surface, one process, one Dockerfile. A React toolchain would add a build
to maintain and buy nothing at this size.

## Testing

`pytest`, with a small fixture index built from a synthetic PDF so tests need no
API key and no 527 MB corpus.

- `test_scoring.py` — tokenisation, overlap, noise penalty, confidence bounds.
- `test_grounding.py` — grounded/ungrounded pairs; the refusal string is always
  considered grounded.
- `test_routing.py` — heuristic classification boundaries. Note: the inherited
  heuristic under-classifies plain causal questions — a bare "Why ...?" scores
  0.28 against a 0.32 threshold and routes as `simple`. The weights are carried
  over unchanged for comparability, so this is covered by a characterisation
  test rather than fixed here. Impact is currently nil because
  `RAG_MODEL_ROUTING_ENABLED` defaults to 0, making the route metadata only.
- `test_retrieval.py` — hybrid merge, ordering, and the fixed candidate gate;
  asserts the no-evidence refusal path is now reachable.
- `test_generation.py` — Groq path with a mocked client; asserts a model refusal
  is **not** overridden by extraction (regression test for bug #1).
- `test_api.py` — `/health`, `/ask`, `/ingest` via `TestClient`.
- `test_ingest.py` — chunk fingerprinting, incremental skip, noise filter.

## Deployment

- `Dockerfile`, single service, uvicorn, non-root user.
- Embedding model baked into the image so `HF_LOCAL_FILES_ONLY=1` holds offline.
- `/health` reports index state and degraded status for container probes.
- `.dockerignore` excludes `data/`, `.venv/`, `venv/`, `__pycache__/`.

## Hygiene

- `.gitignore` covering `.env`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`,
  `data/`, `*.egg-info/`, `.pytest_cache/`.
- `git rm --cached` the 40+ tracked `.pyc` files.
- Restore `.env.example` with every variable documented and no values.
- Pin `requirements.txt`; split out `requirements-dev.txt`.
- Migrate to `langchain-huggingface`'s `HuggingFaceEmbeddings`.
- Rewrite `README.md`: correct setup for both Windows and POSIX, the new module
  map, the honest evaluation methodology, and deployment instructions.

## Risks

**Reported metrics will drop.** This is intended and will be stated plainly in
`report.md`, alongside an explanation of why the previous numbers were invalid.

**Groq as the default requires an API key.** Without one the service falls back
to extraction and logs the downgrade; `/health` reports which path is active, so
the degradation is never silent.

**Ground truth authored via retrieval carries residual circularity.** Expected
answers are written from chunks the current retriever surfaced, so questions it
cannot retrieve at all may get weak ground truth. Mitigated by recording
`expected_sources` at page level, which makes recall measurable independently of
what the retriever returned on the authoring pass. Accepted knowingly as the
cost of the faster path.

**The frontend has no light theme.** Stated above as a deliberate sacrifice.

## Delivery order

1. Hygiene and safety — `.gitignore`, untrack `.pyc`, restore `.env.example`, pin deps.
2. Config, logging, models — the foundation the rest depends on.
3. Module split with behaviour held constant; tests added as each module lands.
4. Bug fixes: refusal override, dead refusal path, silent failures.
5. Groq as primary generation path.
6. Ground truth authoring and the new evaluator.
7. Frontend.
8. Dockerfile, README, final evaluation run and report.
