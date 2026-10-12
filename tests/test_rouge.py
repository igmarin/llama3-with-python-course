import pytest

pytest.importorskip("rouge_score")

from llama_course.rouge import rouge_scores


def test_identical_text_scores_one():
    s = rouge_scores(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert s == pytest.approx({"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0})


def test_disjoint_text_scores_zero():
    assert rouge_scores(["alpha beta"], ["gamma delta"])["rouge1"] == 0.0


def test_partial_overlap_is_between_zero_and_one():
    s = rouge_scores(["the cat sat"], ["the cat ran away"])
    assert 0 < s["rouge2"] < s["rouge1"] < 1


def test_mismatched_or_empty_input_is_rejected():
    with pytest.raises(ValueError):
        rouge_scores(["a"], [])
    with pytest.raises(ValueError):
        rouge_scores([], [])
