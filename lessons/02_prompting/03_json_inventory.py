"""Lesson 2.3: JSON output.

Goal:     Turn free text into data your program can use.
Concept:  ``response_format={"type": "json_object"}`` forces the model to emit
          valid JSON. ``chat_json`` also parses it and raises
          ``InvalidJSONReply`` if the reply was cut off. The input is a
          delivery note from a bike shop's supplier.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/02_prompting/03_json_inventory.py
Expected: {'inner_tubes': 24, 'brake_pads': 60, 'saddles': 7}. The model turns "two
          dozen" into 24 and picks the key spelling itself; small models sometimes
          get a number wrong.
"""

from llama_course.llm import get_llm
from llama_course.structured import chat_json

SYSTEM = (
    "You turn delivery notes into JSON. Use each part name as a key and its quantity "
    "as an integer value, for example {\"chains\": 3}."
)


def main() -> None:
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Delivered today: two dozen inner tubes, sixty brake pads and seven saddles."},
    ]
    print(chat_json(llm, messages, max_tokens=200))


if __name__ == "__main__":
    main()
