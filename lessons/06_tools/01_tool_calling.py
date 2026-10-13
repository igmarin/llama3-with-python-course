"""Lesson 6.1: Let the model call Python functions.

Goal:     Answer questions that need live data or exact arithmetic.
Concept:  A language model guesses text; it cannot look up stock or convert units
          reliably. Tool calling splits the job: the model picks a function and its
          arguments as JSON, your code validates and runs the function, and the model
          turns the result into a sentence.
            question -> model picks {"tool": ..., "arguments": ...}
                     -> Pydantic validates -> your function runs
                     -> model writes the answer from the result
          llama-cpp-python has no tool parser for Llama 3, so the JSON Schema built
          from the tools forces a valid call, and ``ask`` handles the rest. An invalid
          call (wrong name, bad arguments) is sent back to the model once. The model
          never runs code itself: only registered, side-effect-free functions can run.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/06_tools/01_tool_calling.py
Expected: For each question, the tool the model chose, its arguments, the tool
          result and the final answer. The last question needs no tool. A 3B model
          sometimes picks the wrong tool or wrong arguments; the printed call lets you see it.
"""

from llama_course.llm import get_llm
from llama_course.shop_tools import SHOP_TOOLS
from llama_course.tools import ToolCallFailed, ask

QUESTIONS = [
    "How many helmets do you have in stock?",
    "Is the road bike available?",
    "How far is 26.2 miles in kilometers?",
    "What is the shipping cost for a $1,800 order?",
    "What is the shipping cost for a $400 order with express delivery?",
    "Who painted the Mona Lisa?",  # no tool needed
]


def main() -> None:
    llm = get_llm()
    for question in QUESTIONS:
        print(f"Q: {question}")
        try:
            result = ask(llm, SHOP_TOOLS, question, max_tokens=150, temperature=0.0)
        except ToolCallFailed as exc:
            print(f"   Gave up: {exc}\n")
            continue
        if result.tool:
            print(f"   Tool: {result.tool}({result.arguments}) -> {result.result}")
        print(f"A: {result.text}\n")


if __name__ == "__main__":
    main()
