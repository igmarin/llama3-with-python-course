"""Lesson 2.2: Few-shot prompting.

Goal:     Teach the model a task by showing examples.
Concept:  Instead of explaining the task, show a few input/label pairs and leave
          the last label blank. The model copies the pattern. ``max_tokens`` is
          tiny because the label is one word.
Run:      python lessons/02_prompting/02_few_shot.py
Expected: "Positive,"
"""

from llama_course.llm import get_llm
from llama_course.prompts import few_shot_prompt

EXAMPLES = [
    ("I ordered from this place last night, and I'm impressed!", "Positive"),
    ("My order was delayed by over an hour without any updates. Disappointing!", "Negative"),
    ("The food quality is top-notch. Highly recommend!", "Positive"),
]


def main() -> None:
    llm = get_llm()
    prompt = few_shot_prompt(EXAMPLES, "Delicious food, and excellent customer service!")
    output = llm(prompt, max_tokens=2, stop=["Review"])
    print(output["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
