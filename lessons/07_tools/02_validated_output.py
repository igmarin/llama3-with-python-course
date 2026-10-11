"""Lesson 7.2: Validated structured output with Pydantic.

Goal:     Get typed Python objects from the model and catch bad values.
Concept:  Lesson 2.4 constrained JSON with a hand-written schema. A Pydantic model
          is the same schema plus real validation (types, ranges, custom rules) and
          gives you an object instead of a dict. ``chat_model`` sends the model the
          schema, validates the reply, and on failure shows the model its own error
          and asks once more. Schema constraints guide generation; Pydantic is the
          check you can trust.
Run:      python lessons/07_tools/02_validated_output.py
Expected: Three parsed inventories, then the share of the 32 inventory sentences in
          data/inventory_text.csv whose counts all match (compared as sorted lists of
          counts, because the model may name items differently).
"""

import json

from pydantic import BaseModel, Field

from llama_course.data import load_inventory
from llama_course.llm import get_llm
from llama_course.structured import InvalidModelReply, chat_model

SYSTEM = (
    "Extract every item and its count from the text. "
    "Convert number words to digits. Use plural item names."
)


class Item(BaseModel):
    name: str
    count: int = Field(ge=0)


class Inventory(BaseModel):
    items: list[Item]


def extract(llm, text: str) -> Inventory:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return chat_model(llm, messages, Inventory, max_tokens=300, temperature=0.0)


def main() -> None:
    llm = get_llm()
    df = load_inventory()

    for text in df["text"].head(3):
        print(text)
        print("  ->", extract(llm, text).items, "\n")

    correct = failed = 0
    for text, expected in zip(df["text"], df["expected_json"]):
        try:
            got = sorted(i.count for i in extract(llm, text).items)
        except InvalidModelReply:
            failed += 1
            continue
        correct += got == sorted(json.loads(expected).values())
    print(f"Counts match: {correct}/{len(df)} ({correct / len(df):.0%}). Invalid replies: {failed}.")


if __name__ == "__main__":
    main()
