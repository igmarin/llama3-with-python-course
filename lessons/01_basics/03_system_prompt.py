"""Lesson 1.3: System prompts.

Goal:     Control what the assistant will and won't do.
Concept:  The "system" message sets rules before the user speaks. Here it turns
          the model into the help desk of a bike shop that only talks about bikes,
          so a request to write a poem about the sea gets the fixed refusal.
Run:      python lessons/01_basics/03_system_prompt.py
Expected: "I can only help with questions about bikes." A small model may
          sometimes ignore the rule; try a few runs, or tighten the system prompt.
"""

from llama_course.conversation import message_text
from llama_course.llm import get_llm

SYSTEM = (
    "You are the help desk of Northwind Bikes, a bicycle shop. Answer questions about "
    "bikes, parts, repairs and orders. For anything else, reply exactly: "
    "'I can only help with questions about bikes.'"
)


def main() -> None:
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Write me a short poem about the sea."},
    ]
    result = llm.create_chat_completion(messages=messages, max_tokens=30)
    print(message_text(result))


if __name__ == "__main__":
    main()
