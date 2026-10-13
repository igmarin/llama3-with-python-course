"""Lesson 7.1: Prepare training data.

Goal:     Turn the labelled customer messages into the files that LoRA training reads.
Concept:  Fine-tuning shows the model many (input, correct answer) pairs. Each pair
          becomes one short chat: system instruction, the message as the user, and the
          label as the assistant's reply. Three files are needed:
            train.jsonl  examples the model learns from
            valid.jsonl  examples it never learns from, used to watch for memorising
            test.jsonl   the final exam, only used in lesson 7.3
          The test set is the same 33 messages the base model was scored on in lesson 3,
          and it is never used for training or tuning.
Run:      python lessons/07_finetuning/01_prepare_data.py
Expected: Counts per file (111 / 18 / 33) and one example.
"""

import json

from llama_course.finetune import DATA_DIR, prepare_data


def main() -> None:
    counts = prepare_data()
    print(f"Wrote {DATA_DIR}:")
    for name, n in counts.items():
        print(f"  {name}.jsonl: {n} examples")
    first = (DATA_DIR / "train.jsonl").read_text().splitlines()[0]
    print("\nFirst training example:")
    print(json.dumps(json.loads(first), indent=2))


if __name__ == "__main__":
    main()
