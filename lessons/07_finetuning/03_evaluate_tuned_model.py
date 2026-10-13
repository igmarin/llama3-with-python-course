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
Run:      python lessons/07_finetuning/03_evaluate_tuned_model.py
Expected: A table of accuracy for base vs tuned, overall and for easy/hard messages,
          plus the number of replies that did not contain exactly one label.
          Last run on the 3B model: base 84.8%, tuned 97.0% overall; easy messages
          81.5% -> 96.3%, hard messages 100% for both. Only 6 test messages are
          hard and one message is 3 points overall, so a difference of one or two
          messages is noise. Results change with the training settings and seed.
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
    test = load_messages("test")

    print(f"Base model: {profile.mlx_repo}")
    base = score_predictions(test, predict_all(*load(profile.mlx_repo), test["message"]))
    print("Fine-tuned: base + adapter")
    tuned = score_predictions(test, predict_all(*load(profile.mlx_repo, adapter_path=str(ADAPTER_DIR)), test["message"]))

    rows = [("All messages", base["accuracy"], tuned["accuracy"])]
    for difficulty in base["accuracy_by_difficulty"]:
        rows.append((f"  {difficulty}", base["accuracy_by_difficulty"][difficulty], tuned["accuracy_by_difficulty"][difficulty]))
    print(f"\n{'Accuracy':<12}{'Base':>8}{'Tuned':>8}")
    for name, b, t in rows:
        print(f"{name:<12}{b:>8.1%}{t:>8.1%}")
    print(f"\nUnusable replies: base {base['invalid']}, tuned {tuned['invalid']} (of {len(test)})")
    print("\nTuned confusion matrix (rows = true, columns = predicted):")
    print(tuned["confusion"])


if __name__ == "__main__":
    main()
