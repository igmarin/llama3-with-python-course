"""Lesson 2.4: JSON Schema.

Goal:     Fix the exact keys and types of the JSON reply.
Concept:  Lesson 2.3 guaranteed valid JSON but not which keys. A JSON Schema
          constrains the output further, so ``Question`` and ``Answer`` are
          always present and always strings. If ``max_tokens`` is too small the JSON
          is cut off mid-string and ``chat_json`` raises ``InvalidJSONReply``, so
          we ask for a short answer and leave room for it.
Run:      python lessons/02_prompting/04_json_schema.py
Expected: {'Question': '...', 'Answer': '...'} about lists vs tuples (two sentences).
"""

from llama_course.llm import get_llm
from llama_course.structured import chat_json

SCHEMA = {
    "type": "object",
    "properties": {"Question": {"type": "string"}, "Answer": {"type": "string"}},
    "required": ["Question", "Answer"],
}


def main() -> None:
    llm = get_llm()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful tutor for an online education platform. "
            "Answer student questions clearly and concisely.",
        },
        {"role": "user", "content": "What is the difference between a list and a tuple in Python? Answer in two sentences."},
    ]
    print(chat_json(llm, messages, schema=SCHEMA, max_tokens=400))


if __name__ == "__main__":
    main()
