"""Lesson 2.4: JSON Schema.

Goal:     Fix the exact keys and types of the JSON reply.
Concept:  Lesson 2.3 guaranteed valid JSON but not which keys. A JSON Schema
          constrains the output further: here every reply has a ``name`` string,
          a ``price_usd`` number and a ``weight_kg`` number, ready for a product
          catalogue. If ``max_tokens`` is too small the JSON is cut off and
          ``chat_json`` raises ``InvalidJSONReply``, so leave room for it.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/02_prompting/04_json_schema.py
Expected: {'name': 'Ridgeline 29', 'price_usd': 1249, 'weight_kg': 13.8}
"""

from llama_course.llm import get_llm
from llama_course.structured import chat_json

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price_usd": {"type": "number"},
        "weight_kg": {"type": "number"},
    },
    "required": ["name", "price_usd", "weight_kg"],
}

DESCRIPTION = (
    "Meet the Ridgeline 29, our trail hardtail with a 120 mm fork and 12-speed gearing. "
    "It weighs 13.8 kg with pedals and sells for $1,249."
)


def main() -> None:
    llm = get_llm()
    messages = [
        {"role": "system", "content": "Extract the product name, price and weight from the description."},
        {"role": "user", "content": DESCRIPTION},
    ]
    print(chat_json(llm, messages, schema=SCHEMA, max_tokens=100, temperature=0.0))


if __name__ == "__main__":
    main()
