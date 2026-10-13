"""Lesson 7.5: How LoRA settings change the result.

Goal:     Train the same message classifier with four LoRA settings and compare them.
Concept:  LoRA has a few settings you choose before training:
            rank      size of the small matrices it trains (more weights to learn)
            scale     how strongly the adapter's output is added to the model's.
                      PEFT calls this lora_alpha and uses lora_alpha / rank
            dropout   share of adapter values switched off at random while training
            keys      which layers get an adapter: every linear layer, or only the
                      attention projections q_proj and v_proj
            max seq   longest training example, in tokens
          Everything else (data, steps, learning rate) stays the same as lesson 7.2,
          so differences come from these settings alone. Each run is scored on
          validation loss and on the 60-message challenge set.
Requires: pip install -e ".[finetune]" (Apple Silicon only), and lesson 7.1 run first.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/07_finetuning/05_lora_settings.py [--save]
Expected: Four training runs of about a minute each, then a table with trainable
          weights, final validation loss and challenge accuracy per setting.
          ``--save`` writes results_lora_settings.md next to this file.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml
from mlx_lm import load

from llama_course.classify import score_predictions
from llama_course.data import load_messages
from llama_course.finetune import (
    ATTENTION_QV,
    DATA_DIR,
    lora_config,
    mlx_predict,
    parse_training_log,
    peft_scale,
)
from llama_course.llm import ROOT, get_profile

OUT_DIR = ROOT / "adapters" / "lora-settings"

# name: (rank, scale, dropout, keys, max_seq_length)
SETTINGS = {
    "Lesson 7.2 default: rank 8, scale 20, all layers": (8, 20.0, 0.0, None, 512),
    "Rank 12, alpha 8 (scale 0.67), q/v only": (12, peft_scale(8, 12), 0.05, ATTENTION_QV, 512),
    "Rank 2, alpha 4 (scale 2), q/v only, 250 tokens": (2, peft_scale(4, 2), 0.05, ATTENTION_QV, 250),
    "Rank 2, scale 20, q/v only, 250 tokens": (2, 20.0, 0.05, ATTENTION_QV, 250),
}


def train(name: str, repo: str, settings: tuple) -> dict:
    rank, scale, dropout, keys, max_seq = settings
    adapter = OUT_DIR / re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    adapter.mkdir(parents=True, exist_ok=True)
    config = lora_config(repo, adapter, rank, scale, dropout, keys, max_seq, DATA_DIR)
    config_file = adapter / "train_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    print(f"\n== {name}")
    run = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_file)],
        capture_output=True, text=True, check=True,
    )
    log = parse_training_log(run.stdout + run.stderr)
    losses = log["val_loss"]
    last = losses[max(losses)] if losses else float("nan")
    print(f"   trainable weights: {log['trainable_millions']}M, final validation loss: {last:.3f}")
    return {"adapter": adapter, "trainable": log["trainable_millions"], "val_loss": last}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", action="store_true", help="Write results_lora_settings.md.")
    args = parser.parse_args()

    profile = get_profile()
    if not profile.mlx_repo:
        sys.exit(f"Profile {profile.name!r} has no mlx_repo in configs/models.yaml.")
    if not (DATA_DIR / "train.jsonl").exists():
        sys.exit("Run lessons/07_finetuning/01_prepare_data.py first.")

    challenge = load_messages("challenge")
    rows = []
    for name, settings in SETTINGS.items():
        result = train(name, profile.mlx_repo, settings)
        model, tokenizer = load(profile.mlx_repo, adapter_path=str(result["adapter"]))
        accuracy = score_predictions(challenge, mlx_predict(model, tokenizer, challenge["message"]))["accuracy"]
        rows.append((name, result["trainable"], result["val_loss"], accuracy))

    lines = [
        f"# LoRA settings compared: {profile.name}",
        "",
        "| Setting | Trainable weights | Final validation loss | Challenge accuracy (60) |",
        "|---------|-------------------|-----------------------|-------------------------|",
    ]
    lines += [f"| {n} | {t}M | {v:.3f} | {a:.1%} |" for n, t, v, a in rows]
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if args.save:
        out = Path(__file__).parent / "results_lora_settings.md"
        out.write_text(text)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
