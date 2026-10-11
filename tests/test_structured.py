import pytest
from pydantic import BaseModel, Field

from llama_course.structured import (
    InvalidJSONReply,
    InvalidModelReply,
    chat_json,
    chat_model,
)


def test_returns_parsed_object(fake_llm):
    llm = fake_llm(['{"apples": 15}'])
    assert chat_json(llm, [{"role": "user", "content": "x"}]) == {"apples": 15}
    assert llm.calls[0]["response_format"] == {"type": "json_object"}


def test_schema_is_sent_when_given(fake_llm):
    llm = fake_llm(['{"a": "b"}'])
    schema = {"type": "object"}
    chat_json(llm, [], schema=schema, max_tokens=50)
    assert llm.calls[0]["response_format"]["schema"] == schema
    assert llm.calls[0]["max_tokens"] == 50


def test_malformed_json_raises_with_raw_reply(fake_llm):
    llm = fake_llm(['{"apples": 1'])  # cut off
    with pytest.raises(InvalidJSONReply, match="apples"):
        chat_json(llm, [])


class Item(BaseModel):
    name: str
    count: int = Field(ge=0)


class Inventory(BaseModel):
    items: list[Item]


def test_chat_model_returns_a_validated_instance(fake_llm):
    llm = fake_llm(['{"items": [{"name": "apples", "count": 15}]}'])
    inv = chat_model(llm, [{"role": "user", "content": "x"}], Inventory)
    assert inv.items[0] == Item(name="apples", count=15)
    assert llm.calls[0]["response_format"]["schema"] == Inventory.model_json_schema()


def test_chat_model_sends_the_error_back_and_retries(fake_llm):
    llm = fake_llm(['{"items": [{"name": "apples", "count": -1}]}', '{"items": [{"name": "apples", "count": 1}]}'])
    messages = [{"role": "user", "content": "x"}]
    assert chat_model(llm, messages, Inventory).items[0].count == 1
    assert "count" in llm.calls[1]["messages"][-1]["content"]
    assert len(messages) == 1  # caller's list untouched


def test_chat_model_gives_up_after_retries(fake_llm):
    llm = fake_llm(["not json", '{"items": "x"}'])
    with pytest.raises(InvalidModelReply, match="2 attempts"):
        chat_model(llm, [], Inventory, retries=1)
