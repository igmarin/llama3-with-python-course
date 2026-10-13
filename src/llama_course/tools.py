"""Let a model call Python functions safely.

Llama 3 has no built-in tool parser in llama-cpp-python, so we do it in three
steps we control:

1. The model must answer with JSON that names one tool and its arguments. The JSON
   Schema is built from the tools, so the model cannot name a tool that does not exist.
2. Pydantic validates the arguments. If the call is wrong, the error goes back to the
   model for one more try.
3. We run the function ourselves, hand the result to the model, and it writes the answer.

The model never executes anything. Only functions you register can run, and they
should be pure: no shell, no files, no network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from llama_course.structured import InvalidJSONReply, describe_errors

NO_TOOL = "no_tool"
MULTI_TURN_NOTE = (
    "\nYou may call tools one after another. After each tool result, call the next tool "
    f'you need, or reply {{"tool": "{NO_TOOL}"}} once you have every fact the question asks for.'
)


class ToolError(Exception):
    """Raised by a tool for a problem the model should hear about (for example an unknown item)."""


class ToolCallFailed(ValueError):
    """The model could not produce a valid tool call within the allowed retries."""


@dataclass(frozen=True)
class Tool:
    """A function the model may call.

    Args:
        name: Name the model uses in its call.
        description: What the tool does; this is the model's only guide for choosing it.
        args: Pydantic model describing the arguments. Keep it flat (no nested models).
        run: Function that takes the validated arguments and returns a string.
    """

    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[Any], str]


@dataclass(frozen=True)
class ToolAnswer:
    """The outcome of :func:`ask`. ``tool`` is None when the model answered directly.

    ``steps`` lists every ``(tool, arguments, result)`` in order; :func:`ask_multi` fills it.
    """

    text: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result: str | None = None
    steps: tuple[tuple[str, dict[str, Any], str], ...] = ()


class ToolBox:
    """A set of tools plus the schema and prompt that describe them to the model."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {t.name: t for t in tools}

    def schema(self) -> dict[str, Any]:
        """JSON Schema for one call: ``{"tool": <name>, "arguments": {...}}`` or ``no_tool``."""
        options = []
        for tool in self.tools.values():
            args = tool.args.model_json_schema()
            args.pop("title", None)
            options.append(
                {
                    "type": "object",
                    "properties": {"tool": {"const": tool.name}, "arguments": args},
                    "required": ["tool", "arguments"],
                }
            )
        options.append(
            {"type": "object", "properties": {"tool": {"const": NO_TOOL}}, "required": ["tool"]}
        )
        return {"anyOf": options}

    def system_prompt(self) -> str:
        """Instructions listing each tool, its description and its argument schema."""
        lines = [
            "You can call one tool to help answer the user. Reply with JSON only:",
            '{"tool": "<name>", "arguments": {...}}',
            f'If no tool is needed, reply {{"tool": "{NO_TOOL}"}}.',
            "Tools:",
        ]
        for tool in self.tools.values():
            props = tool.args.model_json_schema().get("properties", {})
            lines.append(f"- {tool.name}: {tool.description} Arguments: {list(props)}")
        return "\n".join(lines)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Validate ``arguments`` and run the tool.

        Raises:
            KeyError: Unknown tool name.
            ValidationError: Arguments do not match the tool's model.
        """
        tool = self.tools[name]
        try:
            return tool.run(tool.args.model_validate(arguments))
        except ToolError as exc:
            return f"Error: {exc}"


def ask(
    llm: Any, toolbox: ToolBox, question: str, retries: int = 1, **kwargs: Any
) -> ToolAnswer:
    """Answer ``question``, calling a tool when the model chooses to.

    Args:
        llm: A ``Llama`` instance.
        toolbox: The tools the model may use.
        question: The user's question.
        retries: Extra attempts when the tool call is invalid.
        **kwargs: Passed to ``create_chat_completion`` (for example ``max_tokens``).

    Raises:
        ToolCallFailed: If no valid call is produced within the retries. The message
            holds the last error.
    """
    messages = [
        {"role": "system", "content": toolbox.system_prompt()},
        {"role": "user", "content": question},
    ]
    response_format = {"type": "json_object", "schema": toolbox.schema()}
    error = ""
    for _ in range(retries + 1):
        raw = _content(
            llm.create_chat_completion(messages=messages, response_format=response_format, **kwargs)
        )
        try:
            call = _parse_call(raw)
            if call["tool"] == NO_TOOL:
                return ToolAnswer(_final_answer(llm, question, **kwargs))
            output = toolbox.call(call["tool"], call.get("arguments", {}))
        except (InvalidJSONReply, KeyError, ValidationError) as exc:
            error = _explain(exc, toolbox)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That tool call failed: {error} Try again."},
            ]
            continue
        text = _final_answer(llm, f"Question: {question}\nTool result: {output}", **kwargs)
        return ToolAnswer(text, call["tool"], call.get("arguments", {}), output)
    raise ToolCallFailed(f"No valid tool call after {retries + 1} attempts: {error}")


def _content(result: dict[str, Any]) -> str:
    return result["choices"][0]["message"]["content"]


def _final_answer(llm: Any, content: str, **kwargs: Any) -> str:
    """Get a plain-text reply in a fresh chat.

    The tool-call chat is not reused: its system prompt demands JSON, and the model
    would keep answering in JSON.
    """
    messages = [
        {"role": "system", "content": "Answer the user briefly in plain text."},
        {"role": "user", "content": content},
    ]
    return _content(llm.create_chat_completion(messages=messages, **kwargs)).strip()


def _parse_call(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidJSONReply(f"Not valid JSON: {raw!r}") from exc
    if not isinstance(value, dict):
        raise InvalidJSONReply(f"Expected a JSON object, got: {raw!r}")
    return value


def _explain(exc: Exception, toolbox: ToolBox) -> str:
    if isinstance(exc, KeyError):
        return f"Unknown tool {exc}. Available: {', '.join(toolbox.tools)}."
    if isinstance(exc, ValidationError):
        return f"Bad arguments ({describe_errors(exc)})."
    return f"{exc}."


def ask_multi(
    llm: Any, toolbox: ToolBox, question: str, max_turns: int = 5, **kwargs: Any
) -> ToolAnswer:
    """Answer ``question``, calling as many tools as it needs, up to ``max_turns``.

    Unlike :func:`ask`, which runs at most one tool, this keeps the same chat
    going: after each tool result the model can call another tool or reply
    ``no_tool``. A failed call is explained to the model, as in :func:`ask`, and
    uses up one turn. The final answer sees every tool result, not only the last.

    Args:
        llm: A ``Llama`` instance.
        toolbox: The tools the model may use.
        question: The user's question.
        max_turns: Most model replies before stopping. Prevents endless loops.
        **kwargs: Passed to ``create_chat_completion``.

    Returns:
        A :class:`ToolAnswer` for the last tool that ran (``tool`` is None if none ran).

    Raises:
        ToolCallFailed: If every turn produced an invalid call.
    """
    messages = [
        {"role": "system", "content": toolbox.system_prompt() + MULTI_TURN_NOTE},
        {"role": "user", "content": question},
    ]
    response_format = {"type": "json_object", "schema": toolbox.schema()}
    results: list[tuple[str, dict[str, Any], str]] = []
    error = ""
    for _ in range(max_turns):
        raw = _content(
            llm.create_chat_completion(messages=messages, response_format=response_format, **kwargs)
        )
        try:
            call = _parse_call(raw)
            if call["tool"] == NO_TOOL:
                break
            arguments = call.get("arguments", {})
            output = toolbox.call(call["tool"], arguments)
        except (InvalidJSONReply, KeyError, ValidationError) as exc:
            error = _explain(exc, toolbox)
            reply = f"That tool call failed: {error} Try again."
        else:
            results.append((call["tool"], arguments, output))
            error = ""
            reply = f"Tool result: {output}"
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": reply}]
    if error and not results:
        raise ToolCallFailed(f"No valid tool call in {max_turns} turns: {error}")

    if not results:
        return ToolAnswer(_final_answer(llm, question, **kwargs))
    lines = "\n".join(f"- {tool}({json.dumps(args)}): {output}" for tool, args, output in results)
    text = _final_answer(llm, f"Question: {question}\nTool results:\n{lines}", **kwargs)
    return ToolAnswer(text, *results[-1], steps=tuple(results))
