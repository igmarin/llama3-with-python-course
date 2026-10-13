"""Batch classification: score many messages and write a summary.

Goal:     Classify a large set of messages efficiently and report throughput.
Concept:  Lesson 3 classified messages one at a time. When you have hundreds
          or thousands, you want to know how long it takes and what the
          overall accuracy is. This script loads every message in a split,
          classifies them all, and prints the accuracy, per-class precision
          and recall, and tokens per second.
          For even larger datasets, consider:
          - Running multiple model instances in separate processes.
          - Caching responses for repeated prompts (see docs/howto/cache-responses.md).
          - Using a larger model for the hard messages only.
Read:     README.md in lessons/03_project/ explains the classifier.
Run:      python lessons/03_project/batch_classify.py [--split test|challenge] [--limit N]
Expected: Accuracy, precision/recall per class, throughput in messages per
          second, and total run time.
"""

import argparse
import time

from llama_course.classify import FEW_SHOT_EXAMPLES, predict, score_predictions
from llama_course.data import load_messages
from llama_course.llm import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "challenge"], default="test", help="Which messages to classify.")
    parser.add_argument("--limit", type=int, help="Only classify the first N messages.")
    parser.add_argument("--rules-only", action="store_true", help="Leave the worked examples out of the prompt.")
    args = parser.parse_args()

    df = load_messages(args.split)
    if args.limit:
        df = df.head(args.limit)

    llm = get_llm()
    examples = () if args.rules_only else FEW_SHOT_EXAMPLES

    start = time.perf_counter()
    preds = []
    for message in df["message"]:
        preds.append(predict(llm, message, examples) or "invalid")
    elapsed = time.perf_counter() - start

    result = score_predictions(df, preds)
    print(f"Split: {args.split} ({len(df)} messages)")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Unusable replies: {result['invalid']}")
    print("\nPer-class:")
    for label, m in result["per_class"].items():
        print(f"  {label}: precision {m['precision']:.1%}, recall {m['recall']:.1%} ({m['support']} messages)")
    print("\nBy difficulty:")
    for d, acc in result["accuracy_by_difficulty"].items():
        print(f"  {d}: {acc:.1%}")
    print(f"\nThroughput: {len(df) / elapsed:.1f} messages/s ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()