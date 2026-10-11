"""Lesson 2.1: Prompt formatting.

Goal:     Get a focused answer from plain completion.
Concept:  A completion model continues your text, so structure the text to make
          the answer the natural next thing: an instruction, a question, then
          ``Answer:``. A ``stop`` string ends generation before the model invents
          the next question.
Run:      python lessons/02_prompting/01_prompt_formatting.py
Expected: One or two plain sentences explaining gravity.
"""

from llama_course.llm import get_llm
from llama_course.prompts import qa_prompt


def main() -> None:
    llm = get_llm()
    prompt = qa_prompt("Explain the concept of gravity in simple terms.", "What is gravity?")
    output = llm(prompt, max_tokens=60, stop=["Question:"])
    print(output["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
