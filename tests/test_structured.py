import pytest

from llama_course.structured import InvalidJSONReply, chat_json


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
