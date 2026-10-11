"""Ask the model for JSON and parse it."""

from __future__ import annotations

import json
from typing import Any


class InvalidJSONReply(ValueError):
    """The model's reply was not valid JSON."""


def chat_json(
    llm: Any,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a chat completion in JSON mode and return the parsed object.

    Args:
        llm: A ``Llama`` instance.
        messages: Chat messages to send.
        schema: Optional JSON Schema. When given, the output is constrained to it.
        **kwargs: Passed to ``create_chat_completion`` (for example ``max_tokens``).

    Raises:
        InvalidJSONReply: If the reply cannot be parsed, for example when
            ``max_tokens`` cut it off. The message includes the raw reply.
    """
    response_format: dict[str, Any] = {"type": "json_object"}
    if schema:
        response_format["schema"] = schema
    result = llm.create_chat_completion(
        messages=messages, response_format=response_format, **kwargs
    )
    content = result["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidJSONReply(f"Model did not return valid JSON: {content!r}") from exc
