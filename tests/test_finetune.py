import json

import pandas as pd
import pytest

from llama_course import finetune
from llama_course.data import load_messages
from llama_course.finetune import (
    SYSTEM_PROMPT,
    chat_example,
    parse_label,
    prepare_data,
    prompt_messages,
    split_validation,
)


def test_chat_example_ends_with_the_label_only():
    ex = chat_example("Pay now!", "Repair")
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == SYSTEM_PROMPT
    assert ex["messages"][-1]["content"] == "Repair"


def test_prompt_messages_has_no_answer():
    assert [m["role"] for m in prompt_messages("x")] == ["system", "user"]


def test_validation_split_is_disjoint_stratified_and_repeatable():
    train = load_messages("train")
    t1, v1 = split_validation(train)
    _, v2 = split_validation(train)
    assert v1.equals(v2)
    assert set(t1["message"]).isdisjoint(v1["message"])
    assert len(t1) + len(v1) == len(train)
    assert set(v1["category"]) == set(train["category"]) and (v1["difficulty"] == "hard").any()


def test_validation_never_touches_the_test_set():
    _, valid = split_validation(load_messages("train"))
    assert set(valid["message"]).isdisjoint(load_messages("test")["message"])


def test_prepare_data_writes_three_valid_jsonl_files(tmp_path):
    counts = prepare_data(tmp_path)
    assert counts["test"] == len(load_messages("test"))
    assert counts["train"] + counts["valid"] == len(load_messages("train"))
    for name, n in counts.items():
        lines = (tmp_path / f"{name}.jsonl").read_text().splitlines()
        assert len(lines) == n
        assert json.loads(lines[0])["messages"][-1]["content"] in {"Repair", "Order", "Feedback"}


def test_default_data_dir_is_git_ignored():
    assert finetune.DATA_DIR.name == "finetune_data"
    assert "finetune_data/" in (finetune.ROOT / ".gitignore").read_text()


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Repair", "Repair"),
        ("  order\n", "Order"),
        ("The category is Feedback.", "Feedback"),
        ("Repair or Order", None),  # ambiguous
        ("I don't know", None),
        ("", None),
        ("Repairs are done", None),  # word boundary, not a prefix match
    ],
)
def test_parse_label(reply, expected):
    assert parse_label(reply) == expected


def test_write_jsonl_one_line_per_row(tmp_path):
    df = pd.DataFrame({"message": ["a", "b"], "category": ["Repair", "Order"]})
    finetune.write_jsonl(df, tmp_path / "x.jsonl")
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 2
