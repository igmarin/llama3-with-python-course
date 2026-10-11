"""Multi-turn chat on top of ``Llama.create_chat_completion``."""

from __future__ import annotations

from typing import Any


def message_text(result: dict[str, Any]) -> str:
    """Return the assistant text from a chat-completion result.

    Args:
        result: The dict returned by ``create_chat_completion``.
    """
    return result["choices"][0]["message"]["content"]


class Conversation:
    """A chat that remembers every turn, so follow-up questions have context.

    Args:
        llm: A ``Llama`` instance (see :func:`llama_course.llm.get_llm`).
        system_prompt: Instruction that sets the assistant's behavior. Skipped when empty.
        history: Earlier messages to start from. The list is copied, never modified.

    Example:
        >>> chat = Conversation(llm, system_prompt="Answer in one word.")
        >>> chat.create_completion("Capital of France?")
        'Paris'
    """

    def __init__(
        self,
        llm: Any,
        system_prompt: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> None:
        self.llm = llm
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.history: list[dict[str, str]] = system + list(history or [])

    def create_completion(self, user_prompt: str, **kwargs: Any) -> str:
        """Send a user message, store both sides of the turn, and return the reply.

        Args:
            user_prompt: The next user message.
            **kwargs: Passed to ``create_chat_completion`` (for example ``max_tokens``).
        """
        self.history.append({"role": "user", "content": user_prompt})
        result = self.llm.create_chat_completion(messages=self.history, **kwargs)
        self.history.append(result["choices"][0]["message"])
        return message_text(result)
