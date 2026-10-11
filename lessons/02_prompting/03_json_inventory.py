"""Lesson 2.3: JSON output.

Goal:     Turn free text into data your program can use.
Concept:  ``response_format={"type": "json_object"}`` forces the model to emit
          valid JSON. ``chat_json`` also parses it and raises
          ``InvalidJSONReply`` if the reply was cut off.
Run:      python lessons/02_prompting/03_json_inventory.py
Expected: {'apples': 15, 'oranges': 33, 'potatoes': 5052}. The model converts
          the spelled-out numbers; small models sometimes get one wrong.
"""

from llama_course.llm import get_llm
from llama_course.structured import chat_json

SYSTEM = (
    "You convert inventory lists from text to JSON, extracting item counts and names "
    "from the text as keys and values in the form: item: count; for example, 'banana': 32."
)


def main() -> None:
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Fifteen apples, thirty-three oranges, and five thousand fifty-two potatoes."},
    ]
    print(chat_json(llm, messages, max_tokens=200))


if __name__ == "__main__":
    main()
