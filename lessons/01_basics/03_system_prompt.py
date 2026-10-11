"""Lesson 1.3: System prompts.

Goal:     Control what the assistant will and won't do.
Concept:  The "system" message sets rules before the user speaks. Here it limits
          the assistant to internet-service questions, so a stock question gets
          the refusal text.
Run:      python lessons/01_basics/03_system_prompt.py
Expected: "Sorry, I can't answer that." A small model may sometimes ignore the
          rule; try a few runs, or tighten the system prompt.
"""

from llama_course.conversation import message_text
from llama_course.llm import get_llm

SYSTEM = (
    "You are a helpful and professional customer support assistant for an internet "
    "service provider. If the question or instruction doesn't relate to internet "
    "service, quote the response: 'Sorry, I can't answer that.'"
)


def main() -> None:
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Help me decide which stocks to invest in."},
    ]
    result = llm.create_chat_completion(messages=messages, max_tokens=30)
    print(message_text(result))


if __name__ == "__main__":
    main()
