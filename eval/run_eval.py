"""Checks retrieval quality WITHOUT calling the LLM (free and fast).

For each test question it checks:
  - questions with an expected_source: is that file in the top results?
  - questions with expected_source null (out of scope): is the best score
    below MIN_SCORE, so the assistant would say "I could not find this"?

Run it with:  python -m eval.run_eval
After changing chunk size, MIN_SCORE or the embedding model, run it again.
"""
import json
from pathlib import Path

from app import config
from app.retriever import search


def main() -> None:
    questions = json.loads(
        (Path(__file__).parent / "questions.json").read_text(encoding="utf-8")
    )

    passed = 0
    for item in questions:
        hits = search(item["question"], config.TOP_K)
        best_score = hits[0]["score"]
        sources = [hit["source"] for hit in hits]
        expected = item["expected_source"]

        if expected:
            ok = expected in sources
            detail = f"expected {expected}, got {sources[0]} (best score {best_score})"
        else:
            ok = best_score < config.MIN_SCORE
            detail = f"out of scope, best score {best_score} (limit {config.MIN_SCORE})"

        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {item['question']}\n      {detail}")

    print(f"\n{passed}/{len(questions)} passed")


if __name__ == "__main__":
    main()
