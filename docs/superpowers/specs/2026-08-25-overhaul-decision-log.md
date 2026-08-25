# Overhaul Decision Log

Date: 2026-08-25
Branch: `overhaul/rag-hardening`
Spec: `2026-08-24-airman-rag-overhaul-design.md`
Plan: `../plans/2026-08-24-airman-rag-overhaul.md`

Decisions taken during execution that the code and commit history do not
explain on their own. Recorded so nobody has to re-litigate them.

## The three defects this branch fixed

1. **FAISS never loaded.** The embedding model was absent from the local
   HuggingFace cache while `HF_LOCAL_FILES_ONLY=1`. `SentenceTransformerEmbeddings`
   raised `OSError`, a bare `except Exception:` swallowed it, `vectorstore`
   became `None`, and every query silently ran on keyword matching for the life
   of the process. Import cost 47.9s; it is now 1.4s and `/health` reports
   `vector_loaded`.
2. **The model's refusal was discarded.** `_answer_with_groq_model` returned
   `(REFUSAL_MESSAGE, [])` on a correct refusal; the caller's truthiness test on
   the empty list read that as failure and fell through to extraction, answering
   anyway. Outcomes are now typed: `ANSWERED` / `DECLINED` / `UNAVAILABLE`, and
   only `UNAVAILABLE` may fall back.
3. **The evaluation was circular.** Faithfulness was scored by fuzzy-matching
   answers against the chunks they were extracted from — a check that cannot
   fail. It reported 100% faithfulness and 0% hallucination.

## Decisions

**Plan line numbers were advisory.** The plan's "move this function" tables were
systematically wrong by 2–25 lines; one range pointed at a different function
entirely. Implementers located every function by name instead.

**Routing weights were not retuned.** The heuristic under-classifies plain causal
questions — a bare "Why …?" scores 0.28 against a 0.32 threshold and routes as
`simple`. Keeping the inherited weights preserved comparability with the
benchmark, and the impact is currently nil because `RAG_MODEL_ROUTING_ENABLED`
defaults to `0`, making the route metadata only. Revisit as a separate, measured
change if model routing is ever switched on.

**`RAG_GENERATION_MODE` was left at `extractive`.** The configured
`GROQ_API_KEY` is rejected with HTTP 401. Flipping to `groq` with a dead key
would add a doomed network round trip to every query before falling back. The
Groq path is implemented and unit-tested against a mocked transport but has
never been exercised end-to-end.

**Unanswerable entries were audited against the whole corpus, not top-k
retrieval.** An initial pass marked 16 of 50 questions unanswerable using
`retrieve()` top-k. That conflates "the corpus lacks this" with "retrieval
missed this" — and marking a retrievable-but-missed answer as unanswerable
declares that refusing is *correct*, rewarding the system for a retrieval
failure. Re-auditing across all 6,590 chunks converted 10 entries; 6 remain,
each confirmed genuinely absent. Reported scores went **down** as a result,
which is the intended outcome.

**Faithfulness is reported but labelled not meaningful.** Under extractive
generation every answer is a verbatim span of its cited chunk, so support is
~1.00 by construction and the check cannot fail. Two figures were seen during
development — 44.7% (an artifact of snippet truncation) and 0.0% (an artifact of
circularity). Neither measured hallucination. The report now says so on its face
rather than publishing an unearned 100%, and the qualifier is keyed off the
generation path that actually ran, not off configuration.

**The grounding gate is mode-aware.** The inherited thresholds
(`similarity >= 0.65 AND overlap >= 0.58`) are calibrated for verbatim
extraction. Measured against the 41 corpus-authored reference answers, only
10–11 of 41 pass — so the Groq path, had it been switched on, would have
refused most well-formed paraphrases as ungrounded. A separate abstractive pair
(`0.45 / 0.58`) was calibrated empirically: 32 of 41 reference answers pass,
while invented and off-topic text is still rejected. The extractive pair is
unchanged at `0.65 / 0.58`, so the benchmark cannot move. Only fuzzy similarity
was loosened; token overlap stays at 0.58 as the substance check.

**Git history was not rewritten.** `data/` holds 527 MB across seven PDFs, six
already in history. Purging them would require a force-push on an already-
submitted repository. `data/` is ignored going forward; the committed
vectorstore (16 MB) lets the service run without the PDFs, so deployment is
unaffected.

## Known limitations

- The Groq generation path has never run end-to-end (401 key).
- `app/ingest.py` still reads chunk size and overlap from module constants
  rather than `Settings`; the path settings were unified, the chunking ones were
  not.
- Under the lexical judge fallback, correctness can admit answers that overlap
  the reference vocabulary without answering the question. It reads as a floor.
- The interface has no light theme. This is a deliberate commitment, not an
  omission — see the spec's Night Cockpit section.
