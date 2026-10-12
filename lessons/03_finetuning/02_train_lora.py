"""Lesson 3.2: Fine-tune with LoRA.

Goal:     Train a small add-on ("adapter") that makes Llama better at sorting emails.
Concept:  Full fine-tuning updates every weight, which needs far more memory than a
          laptop has. LoRA freezes the model and trains small extra matrices instead:
          usually well under 1% of the parameters. The result is an adapter of a few
          MB that you load on top of the original model. Here the base model is
          already 4-bit quantized (QLoRA), which is why 3B fits in 16 GB.
          Settings that matter:
            iters         training steps (60 here is about 2 passes over the data).
                          Too few = underfit; too many = memorise.
            batch size    examples per step. Lower it if you run out of memory.
            learning rate step size. LoRA tolerates much more than full fine-tuning,
                          but too high makes the loss jump around.
            mask-prompt   compute the loss on the label only, not on the email text.
          Watch "Val loss" (loss on the validation emails). Here it is lowest after
          about 20 to 50 steps and rises if you train longer (0.05 at step 50, 0.18
          at step 200) while Train loss keeps falling to 0: the model memorises the
          training emails. Choose the number of iterations from validation loss,
          never from the test set, or the test score stops being honest.
Memory:   laptop-16gb  (3B, 4-bit)  about 4 to 6 GB while training.
          studio-128gb (70B, 4-bit) about 45 to 60 GB; raise --batch-size freely.
          These are estimates; check Activity Monitor on your machine.
Requires: pip install -e ".[finetune]"   (Apple Silicon only)
Run:      python lessons/03_finetuning/02_train_lora.py
Expected: Validation loss near 11 at step 1 and about 0.05 to 0.11 afterwards, then
          "Saved adapter" under adapters/. The 3B model takes about a minute on an M-series Mac.
"""

import subprocess
import sys

from llama_course.finetune import DATA_DIR
from llama_course.llm import ROOT, get_profile

ADAPTER_DIR = ROOT / "adapters" / "email-lora"
ITERS = 60
BATCH_SIZE = 4
LEARNING_RATE = 5e-5


def main() -> None:
    profile = get_profile()
    if not profile.mlx_repo:
        sys.exit(f"Profile {profile.name!r} has no mlx_repo in configs/models.yaml.")
    if not (DATA_DIR / "train.jsonl").exists():
        sys.exit("Run lessons/03_finetuning/01_prepare_data.py first.")

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
