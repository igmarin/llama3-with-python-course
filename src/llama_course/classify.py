"""Classify emails into Priority, Updates or Promotions with a Llama model.

The model is asked for JSON that matches a schema whose ``category`` can only be
one of :data:`~llama_course.data.LABELS`, so it cannot invent a new label. If the
reply is still unusable, :func:`classify` returns a default instead of crashing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd

from llama_course.data import LABELS
from llama_course.structured import InvalidJSONReply, chat_json

FALLBACK = "Updates"
INVALID = "invalid"

SYSTEM_PROMPT = (
    "You sort emails into exactly one category.\n"
    "- Priority: the reader must act or respond soon (deadlines, security, payments, appointments).\n"
    "- Updates: information only, no action needed (receipts, newsletters, status, release notes).\n"
    "- Promotions: marketing that tries to sell something (sales, coupons, offers).\n"
    "Judge by what the email asks of the reader, not by urgent-sounding words. "
    'Reply with JSON like {"category": "Updates"}.'
)

SCHEMA = {
    "type": "object",
    "properties": {"category": {"type": "string", "enum": list(LABELS)}},
    "required": ["category"],
}

# Written for this prompt; none of these appear in data/.
FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("Your card was charged twice. Call the bank today so they can reverse one charge.", "Priority"),
    ("Your weekly digest: 5 new articles and 2 replies to your comments.", "Updates"),
    ("Hurry! Take 30% off sneakers until midnight with code RUN30.", "Promotions"),
    ("URGENT notice: the cafeteria is changing its opening hours. No action needed.", "Updates"),
    ("Can you send me the signed form before noon? The office closes at one.", "Priority"),
    ("Last chance: your free upgrade expires tonight. Claim it now!", "Promotions"),
)


def build_messages(
    email: str, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES
) -> list[dict[str, str]]:
    """Build the chat: instructions, worked examples as past turns, then the email."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, label in examples:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps({"category": label})})
    messages.append({"role": "user", "content": email})
    return messages


def predict(llm: Any, email: str, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES) -> str | None:
    """Return the model's label, or ``None`` if the reply is unusable."""
    try:
        reply = chat_json(
            llm, build_messages(email, examples), schema=SCHEMA, max_tokens=30, temperature=0.0
        )
    except InvalidJSONReply:
        return None
    label = reply.get("category")
    return label if label in LABELS else None


def classify(llm: Any, email: str, examples: Sequence[tuple[str, str]] = FEW_SHOT_EXAMPLES) -> str:
    """Return exactly one valid label; falls back to :data:`FALLBACK` on a bad reply."""
    return predict(llm, email, examples) or FALLBACK


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


def evaluate(llm: Any, df: pd.DataFrame) -> dict[str, Any]:
    """Classify every row of ``df`` and score it.

    Args:
        llm: A ``Llama`` instance.
        df: Rows with ``email``, ``category`` and ``difficulty`` columns.

    Returns:
        The :func:`metrics` result plus ``accuracy_by_difficulty`` and ``invalid`` (count).
    """
    preds = [predict(llm, email) or INVALID for email in df["email"]]
    result = metrics(df["category"], preds)
    correct = pd.Series(preds, index=df.index) == df["category"]
    result["accuracy_by_difficulty"] = {
        d: float(correct[df["difficulty"] == d].mean()) for d in sorted(df["difficulty"].unique())
    }
    result["invalid"] = preds.count(INVALID)
    return result
