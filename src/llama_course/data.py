"""Load and validate the course datasets in ``data/``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LABELS = ("Repair", "Order", "Feedback")

_MESSAGE_FILES = {
    "all": "messages.csv",
    "train": "train.csv",
    "test": "test.csv",
    "challenge": "challenge.csv",
}


def load_messages(split: Literal["all", "train", "test", "challenge"] = "all") -> pd.DataFrame:
    """Load the labeled customer messages sent to the Northwind Bikes shop.

    Args:
        split: ``"train"`` and ``"test"`` are a fixed 80/20 split (see
            ``scripts/split_messages.py``); ``"all"`` is every row of those two.
            ``"challenge"`` is a separate set of 60 hard messages, never used for
            training or for choosing a prompt.

    Returns:
        A DataFrame with columns ``message``, ``category`` and ``difficulty``
        (``"easy"`` or ``"hard"``).

    Raises:
        ValueError: If a column is missing, a message is empty, or a label is not
            one of :data:`LABELS`.
    """
    if split not in _MESSAGE_FILES:
        raise ValueError(f"split must be one of {list(_MESSAGE_FILES)}, got {split!r}")
    df = pd.read_csv(DATA_DIR / _MESSAGE_FILES[split])
    missing = {"message", "category", "difficulty"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["message"].isna().any() or (df["message"].str.strip() == "").any():
        raise ValueError("Found an empty message")
    bad = sorted(set(df["category"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels {bad}; expected {list(LABELS)}")
    return df


def load_summaries() -> pd.DataFrame:
    """Load ``text`` / ``summary`` pairs, used as ROUGE references."""
    return pd.read_csv(DATA_DIR / "summaries.csv")


def load_inventory() -> pd.DataFrame:
    """Load delivery notes with ``expected_json`` (a JSON object of part name to quantity)."""
    return pd.read_csv(DATA_DIR / "inventory_text.csv")
