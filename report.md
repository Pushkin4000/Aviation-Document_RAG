# Evaluation Report

Grading method: **lexical-fallback, llm:openai/gpt-oss-120b**

## Methodology

Every question carries a reference answer and source pages authored from the
indexed corpus. Three things are measured separately:

- **Retrieval recall** — did the retriever surface a page that actually
  contains the answer?
- **Correctness** — does the answer convey the reference facts?
- **Faithfulness** — is every claim supported by the chunks the answer
  **cited**? Not by everything retrieved.

That last distinction matters. An earlier version of this report compared the
answer against the chunks it had been copied from, which cannot fail and
reported 100% faithfulness and 0% hallucination. **Those figures were an
artifact of the measurement and are not comparable to the numbers below.**

**On correctness under the lexical fallback.** `judge_method` below tells you
whether each verdict came from an LLM judge or the lexical fallback
(`correct = fact_recall >= 0.5 or similarity >= 0.72`, where similarity is a
token-set ratio). That fallback can score a short answer highly when it is a
near token-subset of a long reference, even if it does not actually answer
the question — this report's own detail rows include `correct=True` verdicts
on `key-fact recall 0.00`. Treat correctness under the lexical fallback as a
**floor, not a precise measure**: it is not more forgiving than the true
answer quality, but it can occasionally admit an answer that merely overlaps
the reference vocabulary. Faithfulness carries the same "structural" caveat
under extractive generation (below); correctness does not have a structural
excuse, only this measurement-fidelity one.

**On the grounding gate.** `is_grounded` applies two threshold pairs. The
strict pair (similarity 0.65, token overlap 0.58) is calibrated for verbatim
extraction. A separate, looser pair applies to answers produced by Groq,
which paraphrases rather than copies.

The abstractive pair was recalibrated on 2026-08-25 against 37 real Groq
answers -- the first run after a working API key arrived. Each answer was
scored against the chunks it actually cited (grounded) and against chunks
retrieved for an unrelated question (ungrounded). Token overlap separates
the two cleanly; fuzzy similarity barely separates them at all:

| | grounded | ungrounded |
|---|---|---|
| token overlap | min 0.24, median 0.55 | median 0.09, **max 0.33** |
| similarity | min 0.48 | max 0.56 |

Overlap 0.35 is therefore the highest threshold admitting **0.0%** of
ungrounded answers, and it admits 81.1% of grounded ones. The previous
value of 0.58 admitted only 37.8%: in the first end-to-end Groq run it
rejected 14 of 50 correct, well-cited answers and eliminated the entire
`reasoning` category. Similarity stays at 0.45 as a weak backstop, since
the measurement shows it does little discriminating work here.

## Dataset
- In-scope questions: 50
- Answered: 38
- Refused: 12
- Out-of-scope questions (must be refused): 15

## Metrics
- Retrieval recall@k (all 50 in-scope questions, 6 of
  which are unanswerable and can never hit): 68.0%
- Retrieval recall@k (answerable subset only, 44 questions —
  the figure that actually measures retrieval, since unanswerable questions
  have no expected source to hit): 77.3%
- Answer correctness: 18.0% — see the lexical-fallback caveat above
- Faithfulness (of answered): 100.0%  — NOT MEANINGFUL under extractive generation (see below)
- Hallucination rate (of answered): 0.0%  — NOT MEANINGFUL under extractive generation (see below)
- Refusal recall (out-of-scope correctly refused): 13/15 (86.7%)
- Refusal precision (of all refusals issued -- 12 in-scope +
  13 out-of-scope -- how many were correct): 14/25 (56.0%)
- Latency p50 / p95: 1153 ms / 1493 ms

### Why faithfulness reads 100.0% here

37 of 38 answered questions were generated extractively (the rest via Groq). An extractive answer is, by construction, a verbatim span copied
out of its cited chunk. Checking such an answer against that same chunk
cannot fail — support is ~1.00 by construction — so the faithfulness and
hallucination figures are structural for those rows, not earned.
Because those rows are the majority here, the headline figure above must not be read as evidence the system does not hallucinate.

Row 28 ("What immediate planning impact does a strong forecast headwind have on PNR calculations?") shows the failure mode this leaves standing: the answer is a faithful, verbatim quote from its cited chunk (`faithful=True`) but does not answer the question (`correct=False`, key-fact recall 0.00, similarity to reference 0.28, token support from cited context 1.00). A faithful quote is not the same thing as a correct answer — closing that gap is what the correctness metric is for.

The metric is informative for rows generated via Groq, where the model is
free to introduce claims absent from the retrieved context. Rows fall back
to extraction when a Groq call cannot complete -- most commonly because the
free tier's tokens-per-day cap has been reached, which returns HTTP 429 and
degrades to extraction by design. If most rows in a run are extractive,
check the run log for rate-limit fallbacks before reading the number.

## Metrics by Question Type
- **applied** (n=20): retrieval 40.0%, correct 15.0%, faithful 100.0%
- **factual** (n=20): retrieval 90.0%, correct 25.0%, faithful 100.0%
- **reasoning** (n=10): retrieval 80.0%, correct 10.0%, faithful 100.0%

## 5 Best Answers
- Q: What does VOR stand for?
  Answer: VOR stands for VHF Omni-directional Range (VOR).
  Expected: VOR stands for VHF Omni-directional Range, a short-range radio navigation aid adopted by ICAO in 1960 that provides bearing information, usually aligned to magnetic north at the VOR station.
  Citations: 11-radio-navigation-2014.pdf (Page 122)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.33, similarity to reference 0.84, token support from cited context 0.86
- Q: What is a flight level?
  Answer: A Flight Level is the aircraft’s height, above 1013.25 hPa, expressed in 100s of feet.
  Expected: A Flight Level is the aircraft's height above the 1013.25 hPa pressure datum, expressed in hundreds of feet; when 1013.25 hPa is set on the altimeter subscale the reading is pressure altitude, and flight levels occur at 500 ft intervals (e.g. FL360 = 36,000 ft).
  Citations: Instruments.pdf (Page 70)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.74, token support from cited context 1.00
- Q: What is the point of no return (PNR) in flight planning?
  Answer: Point of safe return The POINT OF SAFE RETURN (PSR), previously referred to as the POINT OF NO RETURN (PNR), is the furthest point along a planned route to which an aircraft.
  Expected: The Point of No Return (PNR), now often called the Point of Safe Return (PSR), is the furthest point along a planned route from which an aircraft can still fly back to the departure airfield or its alternate within the aircraft's safe endurance.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 265)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.33, similarity to reference 0.84, token support from cited context 1.00
- Q: What does ISA stand for in aviation meteorology?
  Answer: ISA stands for The International Standard Atmosphere (ISA).
  Expected: ISA stands for International Standard Atmosphere, the ICAO-defined average atmosphere used as a common datum for calibrating aircraft pressure instruments and for aircraft design and testing.
  Citations: Meteorology full book.pdf (Page 7)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.33, similarity to reference 0.94, token support from cited context 0.83
- Q: How should pressure altitude be adjusted when QNH differs from standard pressure?
  Answer: On passing the transition altitude the standard pressure setting of 1013.25 hPa is set on the altimeter subscale.
  Expected: Pressure altitude is obtained by setting the altimeter subscale to the standard 1013.25 hPa setting rather than QNH; because the difference between 1013.25 hPa and the actual QNH causes a barometric error of about 30 ft per hPa, the altimeter reading changes accordingly when switching between QNH and the standard setting.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 140)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.79, token support from cited context 1.00

## 5 Worst Answers
- Q: If cumulative delays reduce your fuel reserve near legal minimums, what operational decision should be taken?
  Answer: This information is not available in the provided document(s).
  Expected: If an in-flight fuel check shows that expected fuel remaining on arrival would be less than the required alternate fuel plus final reserve fuel, the commander must weigh traffic and operational conditions at the destination, along the diversion route, and at the destination alternate, when deciding whether to proceed to the destination or divert, so as to land with not less than final reserve fuel.
  Citations: None
  Verdict: correct=False, faithful=True, decision=clarify_low_confidence
  Why: Refused. Incorrect: the corpus does contain this.
- Q: When a VOR radial indication and DME distance trend are inconsistent, what navigation cross-check should be performed?
  Answer: navigation aids alphabetically, together with their individual identifying morse call signs, transmitting frequencies, operational hours, DME aerial elevation, where applicable, and any remarks.
  Expected: (none — unanswerable)
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 16)
  Verdict: correct=False, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.00, token support from cited context 1.00
- Q: What communication and clearance considerations apply before entering controlled Class C airspace?
  Answer: We have seen this horizontal clearance information before when we were discussing multiengine Class B obstacle clearance.
  Expected: (none — unanswerable)
  Citations: 6-mass-and-balance-and-performance-2014.pdf (Page 433)
  Verdict: correct=False, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.00, token support from cited context 1.00
- Q: If a NOTAM closes your planned destination runway, what should be updated in the flight plan?
  Answer: If a pilot lands at an aerodrome other than the destination aerodrome specified in the ICAO flight plan, she must ensure that the ATS unit at the destination is informed within a specified time of her planned ETA at destination.
  Expected: (none — unanswerable)
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 310)
  Verdict: correct=False, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.00, token support from cited context 1.00
- Q: What immediate planning impact does a strong forecast headwind have on PNR calculations?
  Answer: Very strong un-forecast headwinds have been encountered.
  Expected: Any wind component, whether head or tail, reduces the distance to the point of safe return (PNR/PSR) compared with still air, and a stronger wind component produces a greater reduction in that distance; a head and tail wind of the same value reduce the PNR distance by the same amount.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 40)
  Verdict: correct=False, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.28, token support from cited context 1.00
