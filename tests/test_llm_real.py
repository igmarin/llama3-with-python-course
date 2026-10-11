import pytest

from llama_course.llm import get_llm


@pytest.mark.model
def test_real_model_generates_a_token():
    out = get_llm()("Hello", max_tokens=1)
    assert out["choices"][0]["text"] is not None


@pytest.mark.model
def test_same_seed_gives_same_text():
    llm = get_llm()
    kwargs = {"max_tokens": 20, "temperature": 1.0, "seed": 123}
    first = llm("Write one sentence about the sea.", **kwargs)["choices"][0]["text"]
    second = llm("Write one sentence about the sea.", **kwargs)["choices"][0]["text"]
    assert first == second


@pytest.mark.model
def test_token_count_matches_the_tokenizer():
    from llama_course.context import count_tokens

    llm = get_llm()
    assert count_tokens(llm, "Hello world") == len(llm.tokenize(b"Hello world", add_bos=False)) > 0
