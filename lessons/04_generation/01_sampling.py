"""Lesson 4.1: Sampling settings.

Goal:     See how ``temperature`` and ``seed`` change the output.
Concept:  At each step the model gives every possible next token a probability, and
          sampling picks one. ``temperature`` reshapes those probabilities: 0 always
          picks the likeliest token (repeatable), higher values pick unlikely tokens
          more often (varied, riskier). ``top_p`` and ``top_k`` cut off the unlikely
          tail before picking. A fixed ``seed`` makes the random picks repeatable,
          so the same settings give the same text.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/04_generation/01_sampling.py
Expected: Three runs at each temperature (0 gives three identical lines; 1.5 varies
          the most in wording), then two identical lines from the same seed.
"""

from llama_course.conversation import message_text
from llama_course.llm import get_llm

PROMPT = "Write one sentence about a lighthouse."


def complete(llm, **settings) -> str:
    messages = [{"role": "user", "content": PROMPT}]
    return message_text(llm.create_chat_completion(messages=messages, max_tokens=40, **settings)).strip()


def main() -> None:
    llm = get_llm()
    for temperature in (0.0, 0.7, 1.5):
        print(f"\ntemperature={temperature}")
        for _ in range(3):
            print("  ", complete(llm, temperature=temperature))

    print("\nSame seed, temperature=1.0: repeatable")
    for _ in range(2):
        print("  ", complete(llm, temperature=1.0, seed=42))


if __name__ == "__main__":
    main()
