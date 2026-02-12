# Evaluation Report

## Dataset
- Total questions: 50
- Answered (non-refusal): 47
- Refusals: 3

## Metrics
- Retrieval hit-rate: 46/50 (92.0%)
- Faithfulness: 47/47 (100.0% of answered)
- Hallucination rate: 0/47 (0.0% of answered)

## Metrics by Question Type
- **applied**: retrieval 85.0%, faithful 100.0%, hallucination 0.0%
- **factual**: retrieval 95.0%, faithful 100.0%, hallucination 0.0%
- **reasoning**: retrieval 100.0%, faithful 100.0%, hallucination 0.0%

## 5 Best Answers
- Q: What is QNH in altimetry?
  Why: Retrieved chunks matched the query and the final answer stayed grounded with citations.
  Answer: Forecast QNH The lowest forecast QNH within an area, forecast for one hour ahead.
  Citations: Meteorology full book.pdf (Page 136)
- Q: What is QFE in altimetry?
  Why: Retrieved chunks matched the query and the final answer stayed grounded with citations.
  Answer: Altimeter Settings QFE The pressure measured at the aerodrome datum.
  Citations: Meteorology full book.pdf (Page 135)
- Q: What does ISA stand for in aviation meteorology?
  Why: Retrieved chunks matched the query and the final answer stayed grounded with citations.
  Answer: The International Standard Atmosphere (ISA) Because temperature and pressure vary with time and position, both horizontally and vertically, it is necessary, in aviation, to have a standard set of conditions to.
  Citations: Meteorology full book.pdf (Page 12)
- Q: How is dew point defined?
  Why: Retrieved chunks matched the query and the final answer stayed grounded with citations.
  Answer: temperature of 2°C and a dew point of 2°C there must be uniform fog d.
  Citations: Meteorology full book.pdf (Page 533)
- Q: What is a cold front?
  Why: Retrieved chunks matched the query and the final answer stayed grounded with citations.
  Answer: Cold Fronts If cold air is replacing warm air, then the front is called a cold front.
  Citations: Meteorology full book.pdf (Page 317)

## 5 Worst Answers
- Q: What preflight decision is required when expected icing exists and pitot heat is inoperative?
  Why: Refusal was returned. Either retrieval confidence was low or evidence was missing.
  Answer: This information is not available in the provided document(s).
  Citations: None
- Q: If cumulative delays reduce your fuel reserve near legal minimums, what operational decision should be taken?
  Why: Refusal was returned. Either retrieval confidence was low or evidence was missing.
  Answer: This information is not available in the provided document(s).
  Citations: None
- Q: What does VOR stand for?
  Why: Answer appears grounded, but retrieval overlap for the question was weak.
  Answer: VOR) 8VHF Omni-directional Range (VOR) Types of VOR CVOR Conventional VOR is used to define airways and for en-route navigation.
  Citations: 11-radio-navigation-2014.pdf (Page 122)
- Q: When a VOR radial indication and DME distance trend are inconsistent, what navigation cross-check should be performed?
  Why: Answer appears grounded, but retrieval overlap for the question was weak.
  Answer: navigation aids alphabetically, together with their individual identifying morse call signs, transmitting frequencies, operational hours, DME aerial elevation, where applicable, and any remarks.
  Citations: 7-Flight-Planning-and-Monitoring-2014.pdf (Page 16)
- Q: What is load factor in flight mechanics?
  Why: Refusal was returned. Either retrieval confidence was low or evidence was missing.
  Answer: This information is not available in the provided document(s).
  Citations: None
