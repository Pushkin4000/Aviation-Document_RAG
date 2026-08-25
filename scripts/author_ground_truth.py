"""Print top retrieved chunks per question, for authoring ground truth.

This does not write ground truth. It surfaces what the corpus actually says
so a human (or the agent) can write `expected_answer` from real source text
rather than from memory.

Usage: python scripts/author_ground_truth.py > ground_truth_worksheet.txt
"""

import json
from pathlib import Path

from app.engine import get_engine


def main() -> None:
    questions = json.loads(Path("evaluation_set.json").read_text(encoding="utf-8"))
    engine = get_engine()
    for i, item in enumerate(questions, start=1):
        question = item["question"]
        chunks = engine.retrieve(question)[:3]
        print(f"\n{'=' * 78}\n[{i}] ({item['type']}) {question}\n{'=' * 78}")
        for c in chunks:
            print(f"\n--- {c.source} p{c.page} (score {c.score:.3f}) [{c.chunk_id}]")
            print(c.document.page_content[:900])


if __name__ == "__main__":
    main()
