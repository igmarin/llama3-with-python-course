"""Prompt builders for plain-text completion and the Llama 3 chat format."""

from __future__ import annotations

from collections.abc import Sequence


def qa_prompt(instruction: str, question: str) -> str:
    """Build an ``Instruction / Question / Answer`` prompt.

    The prompt ends with ``Answer:`` so the model continues with the answer.
    Pair it with ``stop=["Question:"]`` to keep the model from inventing a next question.
    """
    return f"Instruction: {instruction}\nQuestion: {question}\nAnswer:\n"


def few_shot_prompt(
    examples: Sequence[tuple[str, str]],
    query: str,
    input_label: str = "Input",
    output_label: str = "Label",
) -> str:
    """Build a numbered few-shot prompt; the model completes the last label.

    Args:
        examples: ``(input, label)`` pairs shown to the model.
        query: The new input to label.
        input_label: Name written before each input.
        output_label: Name written before each label.

    Example:
        >>> print(few_shot_prompt([("Flat river path", "Easy")], "Alpine pass", "Route", "Difficulty"))
        Route 1: Flat river path
        Difficulty 1: Easy,
        Route 2: Alpine pass
        Difficulty 2:
    """
    lines = []
    for i, (text, label) in enumerate(examples, start=1):
        lines += [f"{input_label} {i}: {text}", f"{output_label} {i}: {label},"]
    n = len(examples) + 1
    lines += [f"{input_label} {n}: {query}", f"{output_label} {n}:"]
    return "\n".join(lines)


def llama3_chat_prompt(system: str, user: str) -> str:
    """Build a Llama 3 chat prompt by hand.

    ``create_chat_completion`` does this for you. Writing it out once shows what
    the model actually receives. Stop generation on ``<|eot_id|>``.
    The leading ``<|begin_of_text|>`` token is omitted because llama.cpp adds it.
    """
    return (
        f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
