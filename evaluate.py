import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rapidfuzz import fuzz

from app.graph import REFUSAL_MESSAGE, rag_engine

EVAL_FILE = "evaluation_set.json"
DETAIL_FILE = "evaluation_detailed.csv"
REPORT_FILE = "report.md"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def load_questions(file_path: str) -> List[Dict]:
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]


def question_retrieval_hit(question: str, retrieved_chunks: List[Dict]) -> bool:
    if not retrieved_chunks:
        return False
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return False
    coverage_scores = []
    for chunk in retrieved_chunks[:3]:
        content = chunk.get("content", "") or chunk.get("content_snippet", "")
        snippet_tokens = set(tokenize(content))
        overlap = len(q_tokens.intersection(snippet_tokens)) / max(1, len(q_tokens))
        coverage_scores.append(overlap)
    return max(coverage_scores, default=0.0) >= 0.35


def answer_faithful(answer: str, retrieved_chunks: List[Dict]) -> bool:
    if answer == REFUSAL_MESSAGE:
        return True
    context = " ".join(
        (chunk.get("content", "") or chunk.get("content_snippet", "")) for chunk in retrieved_chunks
    ).lower()
    if not context.strip():
        return False
    answer_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if len(sentence.strip()) >= 20
    ]
    if not answer_sentences:
        answer_sentences = [answer]
    for sentence in answer_sentences:
        if fuzz.partial_ratio(sentence.lower(), context) < 60:
            return False
    return True


def explain_case(
    answered: bool, retrieval_hit: bool, faithful: bool, citations: List[str], hallucination: bool
) -> str:
    if not answered:
        return "Refusal was returned. Either retrieval confidence was low or evidence was missing."
    if hallucination:
        return "Answer was generated but not sufficiently supported by retrieved chunks."
    if retrieval_hit and faithful and citations:
        return "Retrieved chunks matched the query and the final answer stayed grounded with citations."
    if faithful and not retrieval_hit:
        return "Answer appears grounded, but retrieval overlap for the question was weak."
    if not citations:
        return "Answer was produced without citations, which weakens traceability."
    return "Mixed quality output: partially grounded but not consistently supported."


def build_report(df: pd.DataFrame) -> str:
    total = len(df)
    answered_count = int(df["answered"].sum())
    retrieval_hits = int(df["retrieval_hit"].sum())
    answered_df = df[df["answered"]]
    faithful_count = int(answered_df["faithful"].sum())
    hallucinations = int(answered_df["hallucination"].sum())

    answered_safe = max(1, answered_count)
    retrieval_hit_rate = (retrieval_hits / max(1, total)) * 100
    faithfulness_rate = (faithful_count / answered_safe) * 100
    hallucination_rate = (hallucinations / answered_safe) * 100

    type_lines = []
    for q_type, group in df.groupby("type"):
        retrieval_rate = float(group["retrieval_hit"].mean())
        answered_group = group[group["answered"]]
        if answered_group.empty:
            faithful_rate = 0.0
            hallucination_rate_group = 0.0
        else:
            faithful_rate = float(answered_group["faithful"].mean())
            hallucination_rate_group = float(answered_group["hallucination"].mean())
        type_lines.append(
            f"- **{q_type}**: retrieval {retrieval_rate*100:.1f}%, "
            f"faithful {faithful_rate*100:.1f}%, hallucination {hallucination_rate_group*100:.1f}%"
        )

    best = df.sort_values(["quality_score", "answered"], ascending=[False, False]).head(5)
    worst = df.sort_values(["quality_score", "answered"], ascending=[True, True]).head(5)

    best_lines = []
    for _, row in best.iterrows():
        best_lines.append(
            f"- Q: {row['question']}\n"
            f"  Why: {row['analysis']}\n"
            f"  Answer: {row['answer']}\n"
            f"  Citations: {row['citations'] or 'None'}"
        )

    worst_lines = []
    for _, row in worst.iterrows():
        worst_lines.append(
            f"- Q: {row['question']}\n"
            f"  Why: {row['analysis']}\n"
            f"  Answer: {row['answer']}\n"
            f"  Citations: {row['citations'] or 'None'}"
        )

    return f"""# Evaluation Report

## Dataset
- Total questions: {total}
- Answered (non-refusal): {answered_count}
- Refusals: {total - answered_count}

## Metrics
- Retrieval hit-rate: {retrieval_hits}/{total} ({retrieval_hit_rate:.1f}%)
- Faithfulness: {faithful_count}/{answered_safe} ({faithfulness_rate:.1f}% of answered)
- Hallucination rate: {hallucinations}/{answered_safe} ({hallucination_rate:.1f}% of answered)

## Metrics by Question Type
{chr(10).join(type_lines) if type_lines else "- No type metrics available"}

## 5 Best Answers
{chr(10).join(best_lines) if best_lines else "- No best cases available"}

## 5 Worst Answers
{chr(10).join(worst_lines) if worst_lines else "- No worst cases available"}
"""


def run_evaluation() -> None:
    questions = load_questions(EVAL_FILE)
    results: List[Dict] = []

    for idx, item in enumerate(questions):
        question = str(item.get("question", "")).strip()
        q_type = str(item.get("type", "unknown")).strip()
        if not question:
            continue

        response = rag_engine.ask(question, debug=True)
        answer = str(response.get("answer", REFUSAL_MESSAGE))
        citations = list(response.get("citations", []))
        retrieved_raw = rag_engine._retrieve(question)  # Internal retrieval for full-text evaluation.
        retrieved_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "score": chunk.score,
                "content": chunk.document.page_content,
            }
            for chunk in retrieved_raw
        ]

        answered = answer != REFUSAL_MESSAGE
        retrieval_hit = question_retrieval_hit(question, retrieved_chunks)
        faithful = answer_faithful(answer, retrieved_chunks)
        hallucination = answered and not faithful

        analysis = explain_case(
            answered=answered,
            retrieval_hit=retrieval_hit,
            faithful=faithful,
            citations=citations,
            hallucination=hallucination,
        )
        quality_score = (
            (1.0 if retrieval_hit else 0.0)
            + (1.0 if faithful else 0.0)
            + (0.25 if answered else 0.0)
            - (1.0 if hallucination else 0.0)
        )

        results.append(
            {
                "id": idx + 1,
                "type": q_type,
                "question": question,
                "answer": answer,
                "citations": "; ".join(citations),
                "retrieved_chunk_ids": "; ".join(
                    chunk.get("chunk_id", "") for chunk in retrieved_chunks[:3]
                ),
                "retrieval_hit": retrieval_hit,
                "faithful": faithful,
                "hallucination": hallucination,
                "answered": answered,
                "quality_score": quality_score,
                "analysis": analysis,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(DETAIL_FILE, index=False)
    Path(REPORT_FILE).write_text(build_report(df), encoding="utf-8")
    print(f"Saved {DETAIL_FILE} and {REPORT_FILE}")


if __name__ == "__main__":
    run_evaluation()
