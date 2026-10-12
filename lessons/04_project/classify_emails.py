"""Lesson 4: Email classifier project.

Goal:     Measure how well a local Llama model sorts emails.
Concept:  Classification with an LLM is a prompt (rules) plus a schema that limits
          the answer to the allowed labels. Judge it on a held-out test set, not on
          a couple of samples, and report easy and hard emails separately.
          Start with the simplest prompt. ``--few-shot`` adds six worked examples,
          and on the training emails they made the model worse (98.4% down to
          88.4%), because a small model copies the pattern of the examples.
          Compare prompts on training data, then score the test set once.
          Run again with another profile to compare models:
              LLAMA_PROFILE=studio-128gb python lessons/04_project/classify_emails.py
Run:      python lessons/04_project/classify_emails.py [--limit N] [--few-shot] [--save]
Expected: Accuracy, per-class precision/recall, accuracy on easy vs hard emails,
          and a confusion matrix. ``--save`` writes results_<profile>.md next to
          this file (results_<profile>_few_shot.md with ``--few-shot``).
"""

import argparse
from pathlib import Path

from llama_course.classify import FEW_SHOT_EXAMPLES, evaluate
from llama_course.data import load_emails
from llama_course.llm import get_llm, get_profile


def report(result: dict, profile: str, n: int, few_shot: bool) -> str:
    """Format evaluation results as Markdown."""
    lines = [
        f"# Email classifier results: {profile}, {'with six worked examples' if few_shot else 'rules only'}",
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
    parser.add_argument("--few-shot", action="store_true", help="Add six worked examples to the prompt.")
    parser.add_argument("--save", action="store_true", help="Write results_<profile>.md.")
    args = parser.parse_args()

    test = load_emails("test")
    if args.limit:
        test = test.head(args.limit)
    profile = get_profile().name
    examples = FEW_SHOT_EXAMPLES if args.few_shot else ()
    text = report(evaluate(get_llm(), test, examples), profile, len(test), args.few_shot)
    print(text)
    if args.save:
        out = Path(__file__).parent / f"results_{profile}{'_few_shot' if args.few_shot else ''}.md"
        out.write_text(text)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
