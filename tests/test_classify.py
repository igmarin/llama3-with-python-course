import json

import pandas as pd
import pytest

from llama_course.classify import (
    FALLBACK,
    FEW_SHOT_EXAMPLES,
    INVALID,
    SCHEMA,
    build_messages,
    classify,
    evaluate,
    metrics,
    predict,
)
from llama_course.data import LABELS, load_messages


def reply(label):
    return json.dumps({"category": label})


def test_valid_reply_is_returned(fake_llm):
    assert classify(fake_llm([reply("Repair")]), "pay now") == "Repair"


def test_schema_only_allows_known_labels():
    assert SCHEMA["properties"]["category"]["enum"] == list(LABELS)


@pytest.mark.parametrize("bad", ['{"category": "Spam"}', "{}", '{"category": "Repair'])
def test_invalid_reply_falls_back_but_predict_reports_none(fake_llm, bad):
    assert classify(fake_llm([bad]), "x") == FALLBACK
    assert predict(fake_llm([bad]), "x") is None


def test_rules_only_prompt_has_no_examples():
    msgs = build_messages("hello", ())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_examples_become_past_turns_before_the_message():
    msgs = build_messages("hello")
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hello"}
    assert len(msgs) == 2 + 2 * len(FEW_SHOT_EXAMPLES)
    assert msgs[2]["content"] == json.dumps({"category": FEW_SHOT_EXAMPLES[0][1]})


def test_few_shot_examples_do_not_leak_from_the_dataset():
    messages = set(load_messages()["message"])
    assert not messages & {text for text, _ in FEW_SHOT_EXAMPLES}


def test_metrics_counts():
    y_true = ["Repair", "Repair", "Order", "Feedback"]
    y_pred = ["Repair", "Order", "Order", INVALID]
    m = metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5
    assert m["per_class"]["Repair"] == {"precision": 1.0, "recall": 0.5, "support": 2}
    assert m["per_class"]["Order"]["precision"] == 0.5
    assert m["per_class"]["Feedback"]["recall"] == 0.0
    assert m["confusion"].loc["Repair", "Order"] == 1
    assert m["confusion"].loc["Feedback", INVALID] == 1


def test_metrics_without_invalid_has_no_invalid_column():
    assert INVALID not in metrics(["Order"], ["Order"])["confusion"].columns


def test_evaluate_splits_by_difficulty_and_counts_invalid(fake_llm):
    df = pd.DataFrame(
        {
            "message": ["a", "b", "c"],
            "category": ["Repair", "Order", "Feedback"],
            "difficulty": ["easy", "easy", "hard"],
        }
    )
    llm = fake_llm([reply("Repair"), reply("Repair"), "not json"])
    r = evaluate(llm, df)
    assert r["accuracy"] == pytest.approx(1 / 3)
    assert r["invalid"] == 1
    assert r["accuracy_by_difficulty"] == {"easy": 0.5, "hard": 0.0}
