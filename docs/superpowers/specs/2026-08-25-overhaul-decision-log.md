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

**`RAG_GENERATION_MODE` was left at `extractive`** while the configured
`GROQ_API_KEY` was rejected with HTTP 401. Superseded — see the addendum
below; a working key arrived and the mode is now `groq`.

**Unanswerable entries were audited against the whole corpus, not top-k
retrieval.** An initial pass marked 16 of 50 questions unanswerable using
`retrieve()` top-k. That conflates "the corpus lacks this" with "retrieval
missed this" — and marking a retrievable-but-missed answer as unanswerable
declares that refusing is *correct*, rewarding the system for a retrieval
failure. Re-auditing across all 6,590 chunks converted 10 entries; 6 remain,
each confirmed genuinely absent. Reported scores went **down** as a result,
which is the intended outcome.

**Faithfulness was reported but labelled not meaningful.** Under extractive
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

---

# Addendum — 2026-08-25, working API key

A valid `GROQ_API_KEY` arrived and the Groq path ran end-to-end for the
first time. Several decisions above were made under the assumption that it
never would; this records what changed and why.

## What the first live run revealed

**The abstractive grounding pair was calibrated on the wrong distribution.**
The pair above (`0.45 / 0.58`) was tuned against the evaluation set's
*reference* answers. That was the best proxy available with a dead key, but
it is not the distribution the model actually produces. The first live run
generated good, correctly cited answers and then discarded 14 of 50 at the
gate — the entire `reasoning` category scored 0.0% correct.

All 13 diagnosable rejections were blocked by **token overlap alone**;
fuzzy similarity passed in every case. The earlier ruling to hold overlap at
0.58 "as the substance check" was simply wrong for paraphrased output.

Recalibrated against 37 real Groq answers, each scored twice — against the
chunks it cited, and against chunks retrieved for an unrelated question:

| | grounded | ungrounded |
|---|---|---|
| token overlap | min 0.24, median 0.55 | median 0.09, **max 0.33** |
| similarity | min 0.48 | max 0.56 |

Overlap separates the two cleanly; similarity barely separates them at all.
`0.35` is the highest overlap threshold admitting **0.0%** of ungrounded
answers, and it admits 81.1% of grounded ones. Replayed over the 37 saved
answers: 23/37 survive at 0.58, 35/37 at 0.35 — 12 rescued, 0 lost.

**Groq retired the Llama 3.x chat models.** `llama-3.3-70b-versatile` and
`llama-3.1-8b-instant` return 404 `model_not_found`. Defaults moved to
`openai/gpt-oss-120b` / `openai/gpt-oss-20b`. `qwen/qwen3.6-27b` was
rejected: it leaks raw `<think>` blocks into `content`. Note that `gpt-oss`
models spend tokens on reasoning before emitting content, so a low
`max_tokens` returns an empty answer — the generator sets no cap, which is
load-bearing, not incidental.

**The original 401 diagnosis was right, but a later 403 was not.** A probe
using raw `urllib` returned HTTP 403 `error code: 1010` — that is
Cloudflare blocking the default User-Agent, not an auth failure. Diagnose
this API through a real client; the bare-`urllib` result is misleading.

## Decisions

**Measured results are reported per generation path, not merged.** Retrieval
is identical across both paths (one retriever); only generation differs.
Correctness moves 18.0% → 48.0%, refusal recall 86.7% → 100%, refusal
precision 56.0% → 73.1%. Latency moves ~1s → ~12s p50, which is the real
cost of the change and is stated alongside the gains.

**Faithfulness is now a real measurement — on the Groq path only.** With
38 of 39 answered rows generated by the model, faithfulness reads 89.7% and
hallucination 10.3%. The "not meaningful" qualifier is no longer blanket:
it keys off the *share* of extractive rows, because a single extractive row
dilutes a figure while a majority manufactures it. Those are different
claims and the report now makes whichever one is true.

**Rate-limit degradation is now visible rather than silent.** The free tier
allows 200,000 tokens/day *per model*; one `evaluate.py` run costs roughly
135,000. Past the cap every call returns 429 and generation falls back to
extraction — which made an exhausted quota read as "the model answered
badly" rather than "the model never ran". Groq calls now retry a brief
per-minute limit and, for the multi-minute daily-cap 429, fall back at once
with an explicit log line rather than stalling the caller. Waiting out a
daily cap in-request would be worse than falling back.

**Evaluation rows are persisted so the report is a pure function of them.**
`evaluate.py --report-only` rebuilds `report.md` from
`evaluation_detailed.csv` + `evaluation_out_of_scope.csv`. Editing the
report's prose should not cost API budget; before this, it did, and that is
how a day's quota got spent re-running a measurement that had not changed.

## Known limitations after this round

- The committed `report.md` is from a **quota-degraded run** (both models'
  daily caps were exhausted) and labels itself as extractive-dominated. The
  Groq figures quoted in the README and above come from the one clean run
  (38/39 rows via Groq, zero rate-limit fallbacks), measured on
  `openai/gpt-oss-20b`. Re-run `evaluate.py` once the cap resets to
  regenerate a matching `report.md`.
- The Groq column was measured on `gpt-oss-20b`, not the configured default
  `gpt-oss-120b`, because the larger model's daily cap was exhausted first.
  The 120b figures are expected to be at least as good, but are unmeasured.
- `HF_LOCAL_FILES_ONLY=0` in the local `.env`, so startup consults the
  HuggingFace Hub. The Dockerfile bakes the model at build time, so `1` is
  correct in-image; the local default is left permissive so a fresh
  checkout works before the cache is warm.

---

# Addendum — landing page

The spec's Night Cockpit section rules that "nothing animates except the
stage indicator". A GSAP landing page was requested for public deployment,
which contradicts that rule head-on. Resolved by splitting the two jobs
rather than compromising either.

**The console did not change.** `web/index.html` still holds to the rule and
moved to `/console`. Motion inside an instrument you are reading is noise,
and that reasoning is unaffected by wanting a landing page.

**`/` is now a separate document that is not an instrument.** It uses scroll
as a descent — the HUD altitude tape unwinds FL410 to ground across the
page, the pipeline draws as a flight plan with an aircraft flying the route,
and the metric dials wind up to measured values. It shares the palette
verbatim so the two pages read as one system.

**Motion is never the only carrier of meaning.** Every entrance is a
`gsap.from()`, so the markup's own state is the final state and a blocked CDN
degrades to a complete static page. The two components that start empty —
dial arcs and comparison bars — have an explicit `applyStaticFallback()`,
which is also the `prefers-reduced-motion` path.

**The landing page reports live health rather than a fixed claim.** Its HUD
status line reads `/health` and shows the real index state and generation
path, so a degraded deployment advertises itself instead of showing a boast
that was true when the HTML was written.

**The published figures include the bad one.** The 10.3% hallucination rate
sits in the instrument panel next to the wins, in refusal red. Omitting it
would have been the natural landing-page move and would have undone the
point of the honest-evaluation work.
