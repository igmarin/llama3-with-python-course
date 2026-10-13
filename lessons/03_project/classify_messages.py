"""Lesson 3: Customer-message classifier project.

Goal:     Measure how well a local Llama model sorts customer messages for a bike
          shop into Repair, Order or Feedback, so each reaches the right person.
Concept:  Classification with an LLM is a prompt (rules plus a few worked
          examples) and a schema that limits the answer to the allowed labels.
          Judge it on a held-out test set, not on a couple of samples, and report
          easy and hard messages separately.
          Examples are a trade-off. On the 129 training messages they raised
          accuracy from 92.2% to 95.3%, but lowered it on hard messages from 88.9%
          to 81.5%. Compare prompts on training data with ``--rules-only``,
          then score the test set once.
          Run again with another profile to compare models:
              LLAMA_PROFILE=laptop-8b python lessons/03_project/classify_messages.py
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/03_project/classify_messages.py [--limit N] [--rules-only] [--save]
Expected: Accuracy, per-class precision/recall, accuracy on easy vs hard messages,
          and a confusion matrix. ``--save`` writes results_<profile>.md next to
          this file (results_<profile>_rules_only.md with ``--rules-only``).
"""

import argparse
from pathlib import Path

from llama_course.classify import FEW_SHOT_EXAMPLES, evaluate
from llama_course.data import load_messages
from llama_course.llm import get_llm, get_profile


def report(result: dict, profile: str, n: int, rules_only: bool) -> str:
    """Format evaluation results as Markdown."""
    lines = [
        f"# Message classifier results: {profile}, {'rules only' if rules_only else 'rules plus six worked examples'}",
        "",
        f"Test messages: {n}. Accuracy: **{result['accuracy']:.1%}**. "
        f"Unusable replies: {result['invalid']}.",
        "",
        "| Class | Precision | Recall | Messages |",
        "|-------|-----------|--------|--------|",
    ]
    for label, m in result["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.1%} | {m['recall']:.1%} | {m['support']} |")
    lines += ["", "| Difficulty | Accuracy |", "|------------|----------|"]
    for d, acc in result["accuracy_by_difficulty"].items():
        lines.append(f"| {d} | {acc:.1%} |")
    lines += ["", "Confusion matrix (rows = true label, columns = predicted):", "", "```"]
    lines += [result["confusion"].to_string(), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="Only classify the first N test messages.")
    parser.add_argument("--rules-only", action="store_true", help="Leave the six worked examples out of the prompt.")
    parser.add_argument("--save", action="store_true", help="Write results_<profile>.md.")
    args = parser.parse_args()

    test = load_messages("test")
    if args.limit:
        test = test.head(args.limit)
    profile = get_profile().name
    examples = () if args.rules_only else FEW_SHOT_EXAMPLES
    text = report(evaluate(get_llm(), test, examples), profile, len(test), args.rules_only)
    print(text)
    if args.save:
        out = Path(__file__).parent / f"results_{profile}{'_rules_only' if args.rules_only else ''}.md"
        out.write_text(text)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
