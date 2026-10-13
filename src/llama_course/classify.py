"""Sort customer messages to a bike shop into Repair, Order or Feedback.

The model is asked for JSON that matches a schema whose ``category`` can only be
one of :data:`~llama_course.data.LABELS`, so it cannot invent a new label. If the
reply is still unusable, :func:`classify` returns a default instead of crashing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from llama_course.data import LABELS
from llama_course.structured import InvalidJSONReply, chat_json

# Used when the model's reply is unusable. In a real shop, route these to a person.
# A silent fallback can misroute messages; the warning makes failures visible.
FALLBACK = "Feedback"
INVALID = "invalid"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You sort customer messages sent to a bike shop into exactly one category.\n"
    "- Repair: the customer wants work done on a bike or part they already have "
    "(fixing, servicing, adjusting, fitting).\n"
    "- Order: the customer wants to buy, reserve, pay, cancel, return or exchange something, "
    "or asks about stock or a delivery, including a problem with an item that just arrived.\n"
    "- Feedback: praise, complaints or suggestions that do not ask the shop to do anything.\n"
    "Judge by what the customer wants now, not by the words they use. "
    'Reply with JSON like {"category": "Order"}.'
)

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": list(LABELS)}},
    "required": ["category"],
}

# Worked examples, used by default. They are written for this prompt and none appear in
# data/. On the 129 training-split messages (3B model) they raised accuracy from 92.2%
# (rules only) to 95.3%, but lowered it on hard messages from 88.9% to 81.5%.
FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("My rear derailleur makes a ticking sound. Can I bring the bike in on Monday?", "Repair"),
    ("Do you have the Cityline helmet in size L? I'd like to buy two.", "Order"),
    ("The new shop sign looks great from the street.", "Feedback"),
    ("Thanks for the fast service! The front brake is squeaking again, can you check it?", "Repair"),
    ("The lights I ordered arrived without batteries. Please send them.", "Order"),
    ("URGENT: your phone number on the website is missing a digit. Just letting you know.", "Feedback"),
)


def build_messages(
    message: str, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES
) -> list[dict[str, str]]:
    """Build the chat: instructions, optional worked examples as past turns, then the message."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, label in examples:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps({"category": label})})
    messages.append({"role": "user", "content": message})
    return messages


def predict(llm: Any, message: str, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES) -> str | None:
    """Return the model's label, or ``None`` if the reply is unusable."""
    try:
        reply = chat_json(
            llm, build_messages(message, examples), schema=SCHEMA, max_tokens=30, temperature=0.0
        )
    except InvalidJSONReply:
        return None
    label = reply.get("category")
    return label if label in LABELS else None


def classify(llm: Any, message: str, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES) -> str:
    """Return exactly one valid label; falls back to :data:`FALLBACK` on a bad reply."""
    label = predict(llm, message, examples)
    if label is None:
        logger.warning("Model reply unusable for message %r; falling back to %s", message, FALLBACK)
    return label or FALLBACK


def metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Compute accuracy, per-class precision/recall, and a confusion matrix.

    Args:
        y_true: Correct labels.
        y_pred: Predicted labels. Use :data:`INVALID` for unusable replies;
            they count as wrong and get their own confusion-matrix column.

    Returns:
        ``accuracy`` (float), ``per_class`` (label -> precision, recall, support)
        and ``confusion`` (DataFrame, rows = true label, columns = predicted).
    """
    t, p = pd.Series(list(y_true), name="true"), pd.Series(list(y_pred), name="predicted")
    columns = list(LABELS) + ([INVALID] if INVALID in set(p) else [])
    confusion = pd.crosstab(t, p).reindex(index=list(LABELS), columns=columns, fill_value=0)
    per_class = {}
    for label in LABELS:
        tp = int(confusion.loc[label, label])
        predicted, actual = int(confusion[label].sum()), int(confusion.loc[label].sum())
        per_class[label] = {
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
            "support": actual,
        }
    return {"accuracy": float((t == p).mean()) if len(t) else 0.0, "per_class": per_class, "confusion": confusion}


def score_predictions(df: pd.DataFrame, preds: Sequence[str]) -> dict[str, Any]:
    """Score predictions (one per row of ``df``, :data:`INVALID` for unusable ones).

    Returns:
        The :func:`metrics` result plus ``accuracy_by_difficulty`` and ``invalid`` (count).
    """
    result = metrics(df["category"], preds)
    correct = pd.Series(list(preds), index=df.index) == df["category"]
    result["accuracy_by_difficulty"] = {
        d: float(correct[df["difficulty"] == d].mean()) for d in sorted(df["difficulty"].unique())
    }
    result["invalid"] = list(preds).count(INVALID)
    return result


def evaluate(
    llm: Any, df: pd.DataFrame, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES
) -> dict[str, Any]:
    """Classify every row of ``df`` with ``llm`` and score it with :func:`score_predictions`.

    Args:
        llm: A ``Llama`` instance.
        df: Rows with ``message``, ``category`` and ``difficulty`` columns.
        examples: Worked examples for the prompt. Pass ``()`` for rules only.
    """
    preds = [predict(llm, message, examples) or INVALID for message in df["message"]]
    return score_predictions(df, preds)
