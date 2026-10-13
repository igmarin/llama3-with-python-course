"""Prepare customer-message data for LoRA fine-tuning with mlx-lm, and read the model's answers.

This module has no MLX import, so it runs and is tested anywhere. The MLX calls live
in the lesson scripts.

Fine-tuning teaches the model our task from examples, so the prompt can be short:
no rules, no worked examples. Each training example is a chat in which the
assistant's reply is the label alone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from llama_course.data import LABELS, load_messages
from llama_course.llm import ROOT

SYSTEM_PROMPT = "Classify the customer message as Repair, Order, or Feedback. Reply with the label only."
DATA_DIR = ROOT / "finetune_data"


def prompt_messages(message: str) -> list[dict[str, str]]:
    """The chat the model sees at training and at test time, without the answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]


def chat_example(message: str, label: str) -> dict[str, Any]:
    """One training example in mlx-lm's ``messages`` format."""
    return {"messages": prompt_messages(message) + [{"role": "assistant", "content": label}]}


def split_validation(
    df: pd.DataFrame, fraction: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a validation set from the training rows.

    Validation loss during training tells you when the model starts to memorise.
    The split keeps every category and difficulty in both parts. The test set is
    never touched.

    Returns:
        ``(train, valid)``.
    """
    valid = df.groupby(["category", "difficulty"], group_keys=False).sample(
        frac=fraction, random_state=seed
    )
    return df.drop(valid.index), valid


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write one ``chat_example`` per row of ``df``."""
    lines = (json.dumps(chat_example(r.message, r.category)) for r in df.itertuples())
    path.write_text("\n".join(lines) + "\n")


def prepare_data(out_dir: Path = DATA_DIR) -> dict[str, int]:
    """Write ``train.jsonl``, ``valid.jsonl`` and ``test.jsonl`` for ``mlx_lm lora``.

    Returns:
        Number of examples per file.
    """
    out_dir.mkdir(exist_ok=True)
    train, valid = split_validation(load_messages("train"))
    parts = {"train": train, "valid": valid, "test": load_messages("test")}
    for name, df in parts.items():
        write_jsonl(df, out_dir / f"{name}.jsonl")
    return {name: len(df) for name, df in parts.items()}


def parse_label(text: str) -> str | None:
    """Find the label in a model reply.

    Returns:
        The label if exactly one distinct label appears (any case, anywhere in the
        text), otherwise ``None``. A reply naming two labels is ambiguous.
    """
    found = {lab for lab in LABELS if re.search(rf"\b{lab}\b", text, re.IGNORECASE)}
    return found.pop() if len(found) == 1 else None


# Training settings shared by lessons 7.2 and 7.5, so both train the same way.
ITERS = 60
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 512
ATTENTION_QV = ("self_attn.q_proj", "self_attn.v_proj")


def peft_scale(alpha: float, rank: int) -> float:
    """Convert PEFT's ``lora_alpha`` to mlx-lm's ``scale``.

    PEFT multiplies the adapter's output by ``lora_alpha / r``. mlx-lm takes that
    multiplier directly as ``scale``, so ``r=12, lora_alpha=8`` in PEFT is
    ``rank=12, scale=0.67`` in mlx-lm.
    """
    return alpha / rank


def lora_config(
    model: str,
    adapter_path: Path,
    rank: int,
    scale: float,
    dropout: float = 0.0,
    keys: Sequence[str] | None = None,
    max_seq_length: int = MAX_SEQ_LENGTH,
    data: Path = DATA_DIR,
) -> dict[str, Any]:
    """Build an ``mlx_lm lora --config`` file as a dict (write it with ``yaml.safe_dump``).

    Args:
        model: Hugging Face repo of the MLX model.
        adapter_path: Where to save the adapter.
        rank: Size of the small matrices LoRA trains. Larger = more weights to train.
        scale: How strongly the adapter's output is added to the model's.
        dropout: Share of adapter values randomly zeroed during training, against memorising.
        keys: Which layers inside each block get an adapter, for example
            :data:`ATTENTION_QV`. ``None`` means every linear layer.
        max_seq_length: Longest training example, in tokens; longer ones are cut.
        data: Folder with ``train.jsonl`` and ``valid.jsonl``.
    """
    params: dict[str, Any] = {"rank": rank, "scale": scale, "dropout": dropout}
    if keys:
        params["keys"] = list(keys)
    return {
        "model": model,
        "train": True,
        "data": str(data),
        "adapter_path": str(adapter_path),
        "fine_tune_type": "lora",
        "num_layers": 16,
        "iters": ITERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": max_seq_length,
        "mask_prompt": True,
        "steps_per_report": 10,
        "steps_per_eval": 20,
        "seed": 0,
        "lora_parameters": params,
    }


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_training_log(text: str) -> dict[str, Any]:
    """Read the trainable-weight count and validation losses from ``mlx_lm lora`` output.

    Returns:
        ``trainable_millions`` (float or None) and ``val_loss`` (dict of step to loss).
    """
    text = _ANSI.sub("", text)
    trainable = re.search(r"Trainable parameters:\s*[\d.]+%\s*\(([\d.]+)M", text)
    val_loss = {
        int(step): float(loss)
        for step, loss in re.findall(r"(?im)^\s*(?:Iter\s+)?(\d+):?\s+val(?:\s+loss)?\s+([\d.]+)", text)
    }
    return {"trainable_millions": float(trainable.group(1)) if trainable else None, "val_loss": val_loss}


def mlx_predict(model: Any, tokenizer: Any, messages: Sequence[str]) -> list[str]:
    """Label each message with an MLX model (greedy), using :func:`parse_label`.

    Returns:
        One label per message, or ``"invalid"`` when the reply has no single label.
    """
    from mlx_lm import generate

    labels = []
    for message in messages:
        prompt = tokenizer.apply_chat_template(
            prompt_messages(message), add_generation_prompt=True, tokenize=False
        )
        labels.append(parse_label(generate(model, tokenizer, prompt=prompt, max_tokens=8)) or "invalid")
    return labels
