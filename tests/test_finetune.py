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


def test_peft_scale_converts_alpha():
    assert finetune.peft_scale(8, 12) == pytest.approx(0.667, abs=0.001)
    assert finetune.peft_scale(4, 2) == 2


def test_lora_config_holds_the_chosen_settings(tmp_path):
    cfg = finetune.lora_config("repo/model", tmp_path / "a", rank=2, scale=2.0, dropout=0.05,
                               keys=finetune.ATTENTION_QV, max_seq_length=250)
    assert cfg["lora_parameters"] == {"rank": 2, "scale": 2.0, "dropout": 0.05,
                                      "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
    assert cfg["max_seq_length"] == 250 and cfg["iters"] == finetune.ITERS and cfg["mask_prompt"]


def test_lora_config_without_keys_targets_every_layer(tmp_path):
    cfg = finetune.lora_config("m", tmp_path, rank=8, scale=20.0)
    assert "keys" not in cfg["lora_parameters"]


def test_parse_training_log_reads_new_and_old_formats():
    new = "Trainable parameters: 0.216% (6.947M/3212.750M)\n     1    val 12.339    3.32s\n\x1b[1m    60\x1b[0m    val 0.001    2.94s\n"
    old = "Iter 1: Val loss 2.500, Val took 1.0s\nIter 60: Val loss 0.300, Val took 1.0s\n"
    assert finetune.parse_training_log(new) == {"trainable_millions": 6.947, "val_loss": {1: 12.339, 60: 0.001}}
    assert finetune.parse_training_log(old)["val_loss"] == {1: 2.5, 60: 0.3}
    assert finetune.parse_training_log("nothing")["trainable_millions"] is None
