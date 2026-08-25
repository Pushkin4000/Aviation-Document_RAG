import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.config import get_settings
from app.engine import get_engine
from app.judge import judge
from app.logging_setup import configure_logging, get_logger
from app.models import REFUSAL_MESSAGE

logger = get_logger("evaluate")

EVAL_FILE = "evaluation_set.json"
OUT_OF_SCOPE_FILE = "out_of_scope_set.json"
DETAIL_FILE = "evaluation_detailed.csv"
REPORT_FILE = "report.md"


def _cited_context(response: Dict) -> str:
    """Text of the chunks the answer actually cited, not everything retrieved."""
    citations = set(response.get("citations", []))
    if not citations:
        return ""
    parts = []
    for chunk in response.get("retrieved_chunks", []) or []:
        if f"{chunk['source']} (Page {chunk['page']})" in citations:
            parts.append(chunk.get("content") or chunk.get("content_snippet", ""))
    return " ".join(parts)


def _retrieval_hit(response: Dict, expected_sources: List[Dict]) -> bool:
    if not expected_sources:
        return False
    got = {(c["source"], int(c["page"])) for c in response.get("retrieved_chunks", []) or []}
    want = {(str(s["source"]), int(s["page"])) for s in expected_sources}
    return bool(got & want)


def evaluate_in_scope(engine, settings) -> List[Dict]:
    questions = json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))
    rows = []
    for idx, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        started = time.perf_counter()
        response = engine.ask(question, debug=True)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = str(response.get("answer", REFUSAL_MESSAGE))
        refused = answer == REFUSAL_MESSAGE
        unanswerable = bool(item.get("unanswerable", False))
        cited = _cited_context(response)

        if refused:
            correct = unanswerable  # refusing an unanswerable question is correct
            faithful = True
            reason = "Refused." + (" Correct: no answer exists in the corpus." if unanswerable
                                   else " Incorrect: the corpus does contain this.")
            method = "rule"
        else:
            verdict = judge(
                question=question,
                answer=answer,
                expected_answer=str(item.get("expected_answer") or ""),
                key_facts=list(item.get("key_facts") or []),
                cited_context=cited,
                settings=settings,
            )
            correct = verdict.correct and not unanswerable
            faithful = verdict.faithful
            reason = verdict.reason
            method = verdict.method

        rows.append({
            "id": idx,
            "type": item.get("type", "unknown"),
            "question": question,
            "expected_answer": item.get("expected_answer") or "",
            "answer": answer,
            "citations": "; ".join(response.get("citations", [])),
            "retrieved_chunk_ids": "; ".join(
                c["chunk_id"] for c in (response.get("retrieved_chunks") or [])[:3]
            ),
            "route": response.get("route", ""),
            "confidence": response.get("confidence", 0.0),
            "decision": response.get("decision", ""),
            "refused": refused,
            "unanswerable": unanswerable,
            "retrieval_hit": _retrieval_hit(response, item.get("expected_sources") or []),
            "correct": correct,
            "faithful": faithful,
            "hallucination": (not refused) and (not faithful),
            "judge_method": method,
            "judge_reason": reason,
            "latency_ms": round(latency_ms, 1),
        })
        logger.info("[%02d/%d] %s -> %s", idx, len(questions), question[:60], response.get("decision"))
    return rows


def evaluate_out_of_scope(engine) -> List[Dict]:
    path = Path(OUT_OF_SCOPE_FILE)
    if not path.exists():
        return []
    rows = []
    for idx, item in enumerate(json.loads(path.read_text(encoding="utf-8")), start=1):
        question = str(item["question"]).strip()
        response = engine.ask(question, debug=True)
        refused = str(response.get("answer", "")) == REFUSAL_MESSAGE
        rows.append({
            "id": idx,
            "question": question,
            "answer": response.get("answer", ""),
            "decision": response.get("decision", ""),
            "confidence": response.get("confidence", 0.0),
            "refused": refused,
        })
    return rows


def build_report(df: pd.DataFrame, oos: pd.DataFrame, method: str) -> str:
    total = len(df)
    answered = df[~df["refused"]]
    n_answered = len(answered)
    retrieval_rate = df["retrieval_hit"].mean() * 100 if total else 0.0
    correctness = df["correct"].mean() * 100 if total else 0.0
    faithful_rate = answered["faithful"].mean() * 100 if n_answered else 0.0
    halluc_rate = answered["hallucination"].mean() * 100 if n_answered else 0.0

    oos_refused = int(oos["refused"].sum()) if len(oos) else 0
    oos_total = len(oos)
    refusal_recall = (oos_refused / oos_total * 100) if oos_total else 0.0

    refusals = df[df["refused"]]
    correct_refusals = int(refusals["unanswerable"].sum()) if len(refusals) else 0
    all_refusals = len(refusals) + oos_total
    all_correct_refusals = correct_refusals + oos_refused
    refusal_precision = (all_correct_refusals / all_refusals * 100) if all_refusals else 0.0

    lat = df["latency_ms"]
    p50 = statistics.median(lat) if total else 0.0
    p95 = sorted(lat)[int(len(lat) * 0.95)] if total > 1 else (lat.iloc[0] if total else 0.0)

    type_lines = []
    for q_type, group in df.groupby("type"):
        ans = group[~group["refused"]]
        type_lines.append(
            f"- **{q_type}** (n={len(group)}): retrieval {group['retrieval_hit'].mean()*100:.1f}%, "
            f"correct {group['correct'].mean()*100:.1f}%, "
            f"faithful {(ans['faithful'].mean()*100 if len(ans) else 0.0):.1f}%"
        )

    ranked = df.assign(
        score=df["correct"].astype(int) * 2 + df["faithful"].astype(int) + df["retrieval_hit"].astype(int)
    )
    best = ranked.sort_values(["score", "confidence"], ascending=[False, False]).head(5)
    worst = ranked.sort_values(["score", "confidence"], ascending=[True, True]).head(5)

    def block(rows):
        return "\n".join(
            f"- Q: {r['question']}\n"
            f"  Answer: {r['answer']}\n"
            f"  Expected: {r['expected_answer'] or '(none — unanswerable)'}\n"
            f"  Citations: {r['citations'] or 'None'}\n"
            f"  Verdict: correct={r['correct']}, faithful={r['faithful']}, decision={r['decision']}\n"
            f"  Why: {r['judge_reason']}"
            for _, r in rows.iterrows()
        )

    return f"""# Evaluation Report

Grading method: **{method}**

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
- In-scope questions: {total}
- Answered: {n_answered}
- Refused: {total - n_answered}
- Out-of-scope questions (must be refused): {oos_total}

## Metrics
- Retrieval recall@k: {retrieval_rate:.1f}%
- Answer correctness: {correctness:.1f}%
- Faithfulness (of answered): {faithful_rate:.1f}%
- Hallucination rate (of answered): {halluc_rate:.1f}%
- Refusal recall (out-of-scope correctly refused): {oos_refused}/{oos_total} ({refusal_recall:.1f}%)
- Refusal precision (refusals that were correct): {refusal_precision:.1f}%
- Latency p50 / p95: {p50:.0f} ms / {p95:.0f} ms

## Metrics by Question Type
{chr(10).join(type_lines) if type_lines else "- none"}

## 5 Best Answers
{block(best)}

## 5 Worst Answers
{block(worst)}
"""


def run_evaluation() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = get_engine()

    state = engine.index_state
    if state.degraded_reason:
        raise SystemExit(
            f"Refusing to evaluate a degraded index: {state.degraded_reason}\n"
            "Results would measure the fallback path, not the system."
        )

    rows = evaluate_in_scope(engine, settings)
    oos_rows = evaluate_out_of_scope(engine)

    df = pd.DataFrame(rows)
    oos = pd.DataFrame(oos_rows)
    df.to_csv(DETAIL_FILE, index=False)

    methods = set(df["judge_method"]) - {"rule"}
    method = ", ".join(sorted(methods)) if methods else "rule-only"
    Path(REPORT_FILE).write_text(build_report(df, oos, method), encoding="utf-8")
    print(f"Saved {DETAIL_FILE} and {REPORT_FILE}")


if __name__ == "__main__":
    run_evaluation()
