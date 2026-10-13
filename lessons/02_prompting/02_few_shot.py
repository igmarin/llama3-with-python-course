"""Lesson 2.2: Few-shot prompting.

Goal:     Teach the model a task by showing examples.
Concept:  Instead of explaining the task, show a few input/label pairs and leave
          the last label blank. The model copies the pattern. Here the task is
          rating how hard a cycling route is.
          Examples show the format, not the full list of allowed answers. The new
          route is harder than every example, so with examples alone the model
          invents a stronger label. One line naming the allowed labels fixes it.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/02_prompting/02_few_shot.py
Expected: "Very Hard," from the examples alone, then "Hard," once the labels are
          listed (3B model, temperature 0).
"""

from llama_course.llm import get_llm
from llama_course.prompts import few_shot_prompt

EXAMPLES = [
    ("Flat paved path along the river, 8 km, lots of benches.", "Easy"),
    ("Rolling country roads with two short climbs, 35 km.", "Moderate"),
    ("Mountain pass with a 12 km climb and steep switchbacks.", "Hard"),
]
NEW_ROUTE = "Rocky single-track descent with drops and loose gravel, 1,200 m of climbing."


def rate(llm, prompt: str) -> str:
    # The stop string "Route" ends generation before the model writes the next
    # example. This works for the 3B and 8B models but may need adjusting for
    # other models that format their output differently.
    output = llm(prompt, max_tokens=3, stop=["Route"], temperature=0.0)
    return output["choices"][0]["text"].strip()


def main() -> None:
    llm = get_llm()
    prompt = few_shot_prompt(EXAMPLES, NEW_ROUTE, input_label="Route", output_label="Difficulty")
    print("Examples only:      ", rate(llm, prompt))
    print("Examples + labels:  ", rate(llm, "Rate each route as Easy, Moderate or Hard.\n" + prompt))


if __name__ == "__main__":
    main()
