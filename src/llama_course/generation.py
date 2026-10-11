"""Streaming chat output."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def stream_chat(llm: Any, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
    """Yield the reply piece by piece as the model generates it.

    Args:
        llm: A ``Llama`` instance.
        messages: Chat messages to send.
        **kwargs: Passed to ``create_chat_completion`` (for example ``max_tokens``).

    Yields:
        Text fragments, usually one token each. Joining them gives the full reply.
    """
    for chunk in llm.create_chat_completion(messages=messages, stream=True, **kwargs):
        piece = chunk["choices"][0]["delta"].get("content")
        if piece:  # the first chunk only carries the role; the last has no content
            yield piece
