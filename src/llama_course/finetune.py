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
