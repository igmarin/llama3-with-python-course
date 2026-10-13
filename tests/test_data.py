import json

import pytest

from llama_course.data import LABELS, load_inventory, load_messages, load_summaries


def test_messages_schema_and_labels():
    df = load_messages()
    assert list(df.columns) == ["message", "category", "difficulty"]
    assert set(df["category"]) == set(LABELS)
    assert set(df["difficulty"]) == {"easy", "hard"}


def test_messages_size_and_no_duplicates():
    df = load_messages()
    assert len(df) >= 150
    assert df["message"].is_unique


def test_classes_are_balanced():
    counts = load_messages()["category"].value_counts()
    assert counts.max() - counts.min() <= 0.05 * counts.sum()


def test_about_a_fifth_are_hard():
    share = (load_messages()["difficulty"] == "hard").mean()
    assert 0.15 <= share <= 0.25


def test_train_and_test_do_not_overlap_and_cover_everything():
    train, test, full = load_messages("train"), load_messages("test"), load_messages()
    assert set(train["message"]).isdisjoint(test["message"])
    assert set(train["message"]) | set(test["message"]) == set(full["message"])


@pytest.mark.parametrize("split", ["train", "test"])
def test_each_split_has_every_label_and_hard_cases(split):
    df = load_messages(split)
    assert set(df["category"]) == set(LABELS)
    assert (df["difficulty"] == "hard").any()


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split"):
        load_messages("validation")


def test_summaries_are_shorter_than_their_text():
    df = load_summaries()
    assert len(df) >= 30
    assert df.notna().all().all()
    assert (df["summary"].str.len() < df["text"].str.len()).all()


def test_inventory_json_is_a_dict_of_positive_ints():
    df = load_inventory()
    assert len(df) >= 30
    for raw in df["expected_json"]:
        data = json.loads(raw)
        assert data and all(isinstance(v, int) and v > 0 for v in data.values())


def test_challenge_set_is_hard_balanced_and_separate():
    challenge = load_messages("challenge")
    assert len(challenge) >= 60
    assert (challenge["difficulty"] == "hard").all()
    counts = challenge["category"].value_counts()
    assert set(counts.index) == set(LABELS)
    assert counts.max() - counts.min() <= 0.1 * len(challenge)
    assert challenge["message"].is_unique
    assert set(challenge["message"]).isdisjoint(load_messages()["message"])
