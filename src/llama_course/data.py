"""Load and validate the course datasets in ``data/``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LABELS = ("Priority", "Updates", "Promotions")

_EMAIL_FILES = {
    "all": "email_categories_data.csv",
    "train": "train.csv",
    "test": "test.csv",
}


def load_emails(split: Literal["all", "train", "test"] = "all") -> pd.DataFrame:
    """Load the labeled email dataset.

    Args:
        split: ``"train"`` and ``"test"`` are a fixed 80/20 split (see
            ``scripts/split_emails.py``); ``"all"`` is every row.

    Returns:
        A DataFrame with columns ``email``, ``category`` and ``difficulty``
        (``"easy"`` or ``"hard"``).

    Raises:
        ValueError: If a column is missing, an email is empty, or a label is not
            one of :data:`LABELS`.
    """
    if split not in _EMAIL_FILES:
        raise ValueError(f"split must be one of {list(_EMAIL_FILES)}, got {split!r}")
    df = pd.read_csv(DATA_DIR / _EMAIL_FILES[split])
    missing = {"email", "category", "difficulty"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["email"].isna().any() or (df["email"].str.strip() == "").any():
        raise ValueError("Found an empty email")
    bad = sorted(set(df["category"]) - set(LABELS))
    if bad:
        raise ValueError(f"Unknown labels {bad}; expected {list(LABELS)}")
    return df


def load_summaries() -> pd.DataFrame:
    """Load ``text`` / ``summary`` pairs, used as ROUGE references."""
    return pd.read_csv(DATA_DIR / "summaries.csv")


def load_inventory() -> pd.DataFrame:
    """Load inventory sentences with ``expected_json`` (a JSON object of item to count)."""
    return pd.read_csv(DATA_DIR / "inventory_text.csv")
