from llama_course.generation import stream_chat


def test_stream_joins_to_the_full_reply(fake_llm):
    llm = fake_llm(["Paris is the capital"])
    pieces = list(stream_chat(llm, [{"role": "user", "content": "?"}], max_tokens=9))
    assert "".join(pieces).strip() == "Paris is the capital"
    assert len(pieces) == 4  # role-only and empty chunks are skipped
    assert llm.calls[0]["stream"] is True and llm.calls[0]["max_tokens"] == 9


def test_stream_is_lazy(fake_llm):
    llm = fake_llm(["a b"])
    stream = stream_chat(llm, [])
    assert llm.calls == []  # nothing sent until we iterate
    next(stream)
    assert len(llm.calls) == 1
