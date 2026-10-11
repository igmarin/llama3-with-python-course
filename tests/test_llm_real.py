import pytest

from llama_course.llm import get_llm


@pytest.mark.model
def test_real_model_generates_a_token():
    out = get_llm()("Hello", max_tokens=1)
    assert out["choices"][0]["text"] is not None
