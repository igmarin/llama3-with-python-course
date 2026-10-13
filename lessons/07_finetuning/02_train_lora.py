"""Lesson 7.2: Fine-tune with LoRA.

Goal:     Train a small add-on ("adapter") that makes Llama better at sorting customer messages.
Concept:  Full fine-tuning updates every weight, which needs far more memory than a
          laptop has. LoRA freezes the model and trains small extra matrices instead:
          usually well under 1% of the parameters. The result is an adapter of a few
          MB that you load on top of the original model. Here the base model is
          already 4-bit quantized (QLoRA), which helps the 3B model train in 16 GB.
          Settings that matter:
            iters         training steps (60 here is about 2 passes over the data).
                          Too few = underfit; too many = memorise.
            batch size    examples per step. Lower it if you run out of memory.
            learning rate step size. LoRA tolerates much more than full fine-tuning,
                          but too high makes the loss jump around.
            mask-prompt   compute the loss on the label only, not on the message text.
          Watch "Val loss" (loss on the validation messages). If it stops falling, or
          rises while Train loss keeps falling, the model is memorising the training
          messages and you should stop earlier. Here it reaches 0.001 by step 60 and
          stays near 0 up to step 200, so 60 steps is enough and cheaper. Choose the
          number of iterations from validation loss, never from the test set, or the
          test score no longer predicts new messages.
Memory:   laptop-16gb  (3B, 4-bit)  about 4 to 6 GB while training.
          studio-128gb (70B, 4-bit) about 45 to 60 GB; raise --batch-size freely.
          These are estimates; check Activity Monitor on your machine.
Requires: pip install -e ".[finetune]"   (Apple Silicon only)
Run:      python lessons/07_finetuning/02_train_lora.py
Expected: Validation loss about 12 at step 1, under 0.5 at steps 20 and 40, and close
          to 0 at step 60, then "Saved adapter" under adapters/. The 3B model takes about a minute on an M-series Mac.
"""

import subprocess
import sys

from llama_course.finetune import DATA_DIR
from llama_course.llm import ROOT, get_profile

ADAPTER_DIR = ROOT / "adapters" / "message-lora"
ITERS = 60
BATCH_SIZE = 4
LEARNING_RATE = 5e-5


def main() -> None:
    profile = get_profile()
    if not profile.mlx_repo:
        sys.exit(f"Profile {profile.name!r} has no mlx_repo in configs/models.yaml.")
    if not (DATA_DIR / "train.jsonl").exists():
        sys.exit("Run lessons/07_finetuning/01_prepare_data.py first.")

    command = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", profile.mlx_repo,
        "--train",
        "--data", str(DATA_DIR),
        "--adapter-path", str(ADAPTER_DIR),
        "--iters", str(ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--mask-prompt",
        "--max-seq-length", "512",
        "--steps-per-report", "10",
        "--steps-per-eval", "20",
    ]
    print("Running:", " ".join(command), "\n")
    subprocess.run(command, check=True)
    print(f"\nSaved adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
