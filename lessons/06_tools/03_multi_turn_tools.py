"""Lesson 6.3: Multi-turn tool use.

Goal:     Answer a question that needs more than one tool call.
Concept:  Lesson 6.1 ran one tool per question. Real tasks often need
          several: check stock for two items, then work out shipping.
          ``ask_multi`` keeps the same chat going after each tool result,
          so the model can call another tool or stop when it has enough.
          The loop runs at most ``max_turns`` times to prevent infinite
          loops if the model keeps calling tools.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/06_tools/03_multi_turn_tools.py
Expected: For each question, the tools the model called (in order), their
          results, and the final answer. A 3B model may not always chain
          tools correctly; the printed trace lets you see where it went
          wrong.
"""

from llama_course.llm import get_llm
from llama_course.shop_tools import SHOP_TOOLS
from llama_course.tools import ToolCallFailed, ask_multi

QUESTIONS = [
    "I need 5 helmets and 3 locks. Are they both in stock?",
    "What is the express shipping cost for a $350 order?",
    "How far is 10 km in miles, and how many mountain bikes do you have?",
]


def main() -> None:
    llm = get_llm()
    for question in QUESTIONS:
        print(f"Q: {question}")
        try:
            result = ask_multi(llm, SHOP_TOOLS, question, max_turns=5, temperature=0.0)
        except ToolCallFailed as exc:
            print(f"   Gave up: {exc}\n")
            continue
        if result.tool:
            print(f"   Last tool: {result.tool}({result.arguments}) -> {result.result}")
        print(f"A: {result.text}\n")


if __name__ == "__main__":
    main()