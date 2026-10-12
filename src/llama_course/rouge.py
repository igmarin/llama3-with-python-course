"""ROUGE scores for generated text such as summaries.

ROUGE compares a generated text with a human reference by overlap:
ROUGE-1 counts shared words, ROUGE-2 shared word pairs, and ROUGE-L the longest
shared word sequence. Scores run from 0 to 1. It rewards matching wording, so a
good summary phrased differently scores lower than it deserves; read it as a
trend across many examples, not a verdict on one.
"""

from __future__ import annotations

from collections.abc import Sequence


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Average ROUGE-1, ROUGE-2 and ROUGE-L F1 over all pairs.

    Raises:
        ValueError: If the two lists differ in length or are empty.
        ImportError: If ``rouge-score`` is missing (``pip install -e ".[finetune]"``).
    """
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must be non-empty and the same length")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        for name, score in scorer.score(ref, pred).items():
            totals[name] += score.fmeasure
    return {name: total / len(predictions) for name, total in totals.items()}
