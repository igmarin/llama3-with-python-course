"""Ask the model for JSON and parse it."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


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


class InvalidModelReply(ValueError):
    """The model still gave an invalid reply after the allowed retries."""


def describe_errors(exc: ValidationError) -> str:
    """Summarise a Pydantic error in one line a model can act on."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}" for e in exc.errors()
    )


def chat_model(
    llm: Any,
    messages: list[dict[str, str]],
    model: type[T],
    retries: int = 1,
    **kwargs: Any,
) -> T:
    """Ask for JSON that matches a Pydantic model and return a validated instance.

    The model's JSON Schema constrains generation, and Pydantic checks the result
    again (types, ranges, custom rules). If validation fails, the error is sent
    back to the model and it gets another try.

    Args:
        llm: A ``Llama`` instance.
        messages: Chat messages to send. The list is not modified.
        model: A Pydantic ``BaseModel`` subclass describing the reply.
        retries: Extra attempts after the first.
        **kwargs: Passed to ``create_chat_completion`` (for example ``max_tokens``).

    Raises:
        InvalidModelReply: If every attempt fails; the message lists the last errors.
    """
    messages = list(messages)
    response_format = {"type": "json_object", "schema": model.model_json_schema()}
    last: ValidationError | None = None
    for _ in range(retries + 1):
        result = llm.create_chat_completion(
            messages=messages, response_format=response_format, **kwargs
        )
        content = result["choices"][0]["message"]["content"]
        try:
            return model.model_validate_json(content)
        except ValidationError as exc:
            last = exc
            messages += [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That reply is invalid ({describe_errors(exc)}). "
                    "Reply again with corrected JSON only.",
                },
            ]
    raise InvalidModelReply(f"Invalid after {retries + 1} attempts: {describe_errors(last)}")
