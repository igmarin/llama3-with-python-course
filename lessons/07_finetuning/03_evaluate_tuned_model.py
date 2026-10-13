"""Lesson 7.3: Did fine-tuning help?

Goal:     Compare the base model and the fine-tuned model on messages neither has seen.
Concept:  Training loss going down proves nothing about new messages. The real
          check is the held-out test set, scored the same way for both models.
          Both get the same short prompt, with no examples; the fine-tuned one
          has learned the task from data instead of from the prompt.
          Read the "hard" row first: easy messages are usually solved by both.
          Keep the comparison in perspective: the prompt from lesson 3 (rules,
          examples and a schema) already scores 100% on this test set. Fine-tuning
          pays off when a prompt gets long, slow or fragile, or when the task has
          no short rules.
Requires: pip install -e ".[finetune]" and lessons 7.1 and 7.2 run first.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/07_finetuning/03_evaluate_tuned_model.py
Expected: Accuracy for base vs tuned on the test set and on the challenge set, plus
          the number of replies that did not contain exactly one label.
          Last run on the 3B model: test set 84.8% -> 97.0%; challenge set (60 hard
          messages) 70.0% -> 93.3%. One test message is 3 points, so a difference
          of one or two messages is noise. Results change with settings and seed.
"""

import sys

from mlx_lm import generate, load

from llama_course.classify import INVALID, score_predictions
from llama_course.data import load_messages
from llama_course.finetune import parse_label, prompt_messages
from llama_course.llm import ROOT, get_profile

ADAPTER_DIR = ROOT / "adapters" / "message-lora"


def predict_all(model, tokenizer, messages) -> list[str]:
    predictions = []
    for message in messages:
        prompt = tokenizer.apply_chat_template(
            prompt_messages(message), add_generation_prompt=True, tokenize=False
        )
        reply = generate(model, tokenizer, prompt=prompt, max_tokens=8)  # greedy by default
        predictions.append(parse_label(reply) or INVALID)
    return predictions


def main() -> None:
    profile = get_profile()
    if not ADAPTER_DIR.exists():
        sys.exit("No adapter found. Run lessons/07_finetuning/02_train_lora.py first.")
    splits = {name: load_messages(name) for name in ("test", "challenge")}
    models = {
        "Base": load(profile.mlx_repo),
        "Tuned": load(profile.mlx_repo, adapter_path=str(ADAPTER_DIR)),
    }
    print(f"Base model: {profile.mlx_repo}; tuned = base + adapter")

    for split, df in splits.items():
        base, tuned = (score_predictions(df, predict_all(*models[m], df["message"])) for m in models)
        rows = [("All messages", base["accuracy"], tuned["accuracy"])]
        for difficulty in base["accuracy_by_difficulty"]:
            rows.append((f"  {difficulty}", base["accuracy_by_difficulty"][difficulty], tuned["accuracy_by_difficulty"][difficulty]))
        print(f"\n{split.capitalize()} set ({len(df)} messages)")
        print(f"{'Accuracy':<14}{'Base':>8}{'Tuned':>8}")
        for name, b, t in rows:
            print(f"{name:<14}{b:>8.1%}{t:>8.1%}")
        print(f"Unusable replies: base {base['invalid']}, tuned {tuned['invalid']}")
        print("Tuned confusion matrix (rows = true, columns = predicted):")
        print(tuned["confusion"])


if __name__ == "__main__":
    main()
