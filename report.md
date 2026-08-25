# Evaluation Report

Grading method: **lexical-fallback**

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

## Dataset
- In-scope questions: 50
- Answered: 38
- Refused: 12
- Out-of-scope questions (must be refused): 15

## Metrics
- Retrieval recall@k: 68.0%
- Answer correctness: 20.0%
- Faithfulness (of answered): 55.3%
- Hallucination rate (of answered): 44.7%
- Refusal recall (out-of-scope correctly refused): 13/15 (86.7%)
- Refusal precision (refusals that were correct): 55.6%
- Latency p50 / p95: 200 ms / 492 ms

## Metrics by Question Type
- **applied** (n=20): retrieval 40.0%, correct 15.0%, faithful 53.3%
- **factual** (n=20): retrieval 90.0%, correct 30.0%, faithful 50.0%
- **reasoning** (n=10): retrieval 80.0%, correct 10.0%, faithful 66.7%

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
  Why: key-fact recall 0.00, similarity to reference 0.74, token support from cited context 0.80
- Q: What is the point of no return (PNR) in flight planning?
  Answer: Point of safe return The POINT OF SAFE RETURN (PSR), previously referred to as the POINT OF NO RETURN (PNR), is the furthest point along a planned route to which an aircraft.
  Expected: The Point of No Return (PNR), now often called the Point of Safe Return (PSR), is the furthest point along a planned route from which an aircraft can still fly back to the departure airfield or its alternate within the aircraft's safe endurance.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 265)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.33, similarity to reference 0.84, token support from cited context 1.00
- Q: How should pressure altitude be adjusted when QNH differs from standard pressure?
  Answer: On passing the transition altitude the standard pressure setting of 1013.25 hPa is set on the altimeter subscale.
  Expected: Pressure altitude is obtained by setting the altimeter subscale to the standard 1013.25 hPa setting rather than QNH; because the difference between 1013.25 hPa and the actual QNH causes a barometric error of about 30 ft per hPa, the altimeter reading changes accordingly when switching between QNH and the standard setting.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 140)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.79, token support from cited context 1.00
- Q: How do pressure-gradient force and Coriolis force jointly determine large-scale wind direction?
  Answer: • With the geostrophic wind the pressure gradient force is equal to the Coriolis force.
  Expected: The pressure gradient force drives air from high to low pressure and determines wind speed, being stronger where isobars are closely spaced, while the Coriolis force, caused by the Earth's rotation, deflects the moving air; at equilibrium (the geostrophic wind) the two forces balance and the resulting wind blows parallel to the isobars, with low pressure to the left of the wind in the Northern Hemisphere (Buys Ballot's Law).
  Citations: Meteorology full book.pdf (Page 160)
  Verdict: correct=True, faithful=True, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.87, token support from cited context 1.00

## 5 Worst Answers
- Q: What immediate planning impact does a strong forecast headwind have on PNR calculations?
  Answer: Very strong un-forecast headwinds have been encountered.
  Expected: Any wind component, whether head or tail, reduces the distance to the point of safe return (PNR/PSR) compared with still air, and a stronger wind component produces a greater reduction in that distance; a head and tail wind of the same value reduce the PNR distance by the same amount.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 40)
  Verdict: correct=False, faithful=False, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.28, token support from cited context 0.00
- Q: What handling consequence is expected if the aircraft center of gravity moves beyond the aft limit?
  Answer: If the centre of gravity moves aft, the strength of the lift/weight couple is decreased because the arm.
  Expected: With the CG outside the aft limit, longitudinal stability is reduced and, if far enough aft, the aircraft can become very unstable; stick forces in pitch become light, risking overstress by applying excessive g; recovery from a spin may be more difficult because a flat spin is more likely to develop; and range and endurance will probably decrease due to the extra drag from the extreme manoeuvres needed to control the aircraft.
  Citations: 6-mass-and-balance-and-performance-2014.pdf (Page 241)
  Verdict: correct=False, faithful=False, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.40, token support from cited context 0.27
- Q: How does a higher-than-ISA temperature affect takeoff distance planning?
  Answer: • The error is proportional to the difference between actual and ISA temperature, and the vertical distance of the.
  Expected: Higher-than-ISA temperature reduces air density, which increases takeoff distance in three ways: reduced density lowers engine thrust and power so acceleration is less; it raises the true airspeed needed for a given indicated take-off safety speed, requiring more distance to reach it; and with less thrust available the initial climb angle is reduced, requiring a longer horizontal distance to reach screen height.
  Citations: Meteorology full book.pdf (Page 138)
  Verdict: correct=False, faithful=False, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.36, token support from cited context 0.33
- Q: If the static source is blocked and the pitot source is clear, what errors appear on the altimeter and VSI?
  Answer: If altitude is changed with a blocked pitot head and clear static source, the IAS will decrease during a descent because the pressure locked inside the capsule remains constant while the static pressure of the air surrounding the capsule increases.
  Expected: With the static source blocked and pitot clear, the altimeter freezes at the height it read when the blockage occurred and will not register further change: if the aircraft climbs, the altimeter under-reads (shows lower than actual height), and if it descends, the altimeter over-reads (shows higher than actual height); the error grows the further the aircraft moves from the blockage height.
  Citations: Instruments.pdf (Page 51)
  Verdict: correct=False, faithful=False, decision=answer
  Why: key-fact recall 0.00, similarity to reference 0.58, token support from cited context 0.37
- Q: If cumulative delays reduce your fuel reserve near legal minimums, what operational decision should be taken?
  Answer: This information is not available in the provided document(s).
  Expected: If an in-flight fuel check shows that expected fuel remaining on arrival would be less than the required alternate fuel plus final reserve fuel, the commander must weigh traffic and operational conditions at the destination, along the diversion route, and at the destination alternate, when deciding whether to proceed to the destination or divert, so as to land with not less than final reserve fuel.
  Citations: None
  Verdict: correct=False, faithful=True, decision=clarify_low_confidence
  Why: Refused. Incorrect: the corpus does contain this.
