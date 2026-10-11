"""Count tokens and keep a chat inside the model's context window.

The context window (``n_ctx``) is the most text, in tokens, the model can read
at once: the prompt plus the reply it generates. A long chat eventually exceeds it.
"""

from __future__ import annotations

from typing import Any

# Each chat message costs a few extra tokens for the role header and end marker.
# 8 is a safe estimate for Llama 3; other models differ slightly.
MESSAGE_OVERHEAD = 8


class ContextTooLong(ValueError):
    """The system prompt and the newest message alone do not fit the budget."""


def count_tokens(llm: Any, text: str) -> int:
    """Return how many tokens ``text`` uses for this model."""
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def messages_tokens(llm: Any, messages: list[dict[str, str]]) -> int:
    """Estimate the tokens a chat uses, including per-message overhead."""
    return sum(count_tokens(llm, m["content"]) + MESSAGE_OVERHEAD for m in messages)


def trim_history(
    llm: Any, messages: list[dict[str, str]], max_tokens: int
) -> list[dict[str, str]]:
    """Drop the oldest turns until the chat fits in ``max_tokens``.

    System messages and the newest message are always kept. The remaining history
    starts with a user message, which chat templates expect. The input list is
    not modified.

    Args:
        llm: A ``Llama`` instance (used to count tokens).
        messages: The full chat; the last item is the new user message.
        max_tokens: Budget for the prompt. Leave room for the reply, for example
            ``n_ctx - max_reply_tokens``.

    Raises:
        ContextTooLong: If the system messages plus the newest message exceed the budget.
    """
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    newest = rest.pop()
    if messages_tokens(llm, system + [newest]) > max_tokens:
        raise ContextTooLong(
            f"System prompt and newest message need more than {max_tokens} tokens."
        )
    while rest and messages_tokens(llm, system + rest + [newest]) > max_tokens:
        rest.pop(0)
    while rest and rest[0]["role"] != "user":  # never start with a dangling reply
        rest.pop(0)
    return system + rest + [newest]
