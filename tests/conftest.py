import importlib.util
import os
from pathlib import Path

import pytest

from llama_course.llm import get_profile


def pytest_collection_modifyitems(items):
    path = Path(os.environ.get("LLAMA_MODEL_PATH") or get_profile().path)
    reason = None
    if importlib.util.find_spec("llama_cpp") is None:
        reason = "llama_cpp is not installed"
    elif not path.is_file():
        reason = f"model file missing: {path}"
    if reason:
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if "model" in item.keywords:
                item.add_marker(skip)


class FakeLLM:
    """Stands in for ``Llama``: returns canned replies and records each call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        reply = self.replies.pop(0)
        if kwargs.get("stream"):
            return self._stream(reply)
        return {"choices": [{"message": {"role": "assistant", "content": reply}}]}

    @staticmethod
    def _stream(reply):
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        for word in reply.split(" "):
            yield {"choices": [{"delta": {"content": word + " "}}]}
        yield {"choices": [{"delta": {}}]}

    def tokenize(self, data, add_bos=True):
        return data.decode().split()  # one token per word


@pytest.fixture
def fake_llm():
    return FakeLLM
