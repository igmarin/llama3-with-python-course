"""Lesson 4: Email classifier project.

Goal:     Measure how well a local Llama model sorts emails.
Concept:  Classification with an LLM is a prompt (rules + examples) plus a schema
          that limits the answer to the allowed labels. Judge it on a held-out
          test set, not on a couple of samples, and report easy and hard emails
          separately. Run it again with another profile to compare models:
              LLAMA_PROFILE=studio-128gb python lessons/04_project/classify_emails.py
Run:      python lessons/04_project/classify_emails.py [--limit N] [--save]
Expected: Accuracy, per-class precision/recall, accuracy on easy vs hard emails,
          and a confusion matrix. ``--save`` writes results_<profile>.md next to
          this file.
"""

import argparse
from pathlib import Path

from llama_course.classify import evaluate
from llama_course.data import load_emails
from llama_course.llm import get_llm, get_profile


def report(result: dict, profile: str, n: int) -> str:
    """Format evaluation results as Markdown."""
    lines = [
        f"# Email classifier results: {profile}",
        "",
        f"Test emails: {n}. Accuracy: **{result['accuracy']:.1%}**. "
        f"Unusable replies: {result['invalid']}.",
        "",
        "| Class | Precision | Recall | Emails |",
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
    parser.add_argument("--limit", type=int, help="Only classify the first N test emails.")
    parser.add_argument("--save", action="store_true", help="Write results_<profile>.md.")
    args = parser.parse_args()

    test = load_emails("test")
    if args.limit:
        test = test.head(args.limit)
    profile = get_profile().name
    text = report(evaluate(get_llm(), test), profile, len(test))
    print(text)
    if args.save:
        out = Path(__file__).parent / f"results_{profile}.md"
        out.write_text(text)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
