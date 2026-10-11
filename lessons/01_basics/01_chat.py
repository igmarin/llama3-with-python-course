"""Lesson 1.1: Ask Llama a question.

Goal:     Load a model and get one answer.
Concept:  ``llm(prompt)`` is plain text completion. The model continues the text
          you give it. It does not know it is in a chat, so the reply may ramble
          or repeat. Lesson 1.2 fixes that with chat messages.
Run:      python lessons/01_basics/01_chat.py
Expected: Text about databases that may ramble until it hits ``max_tokens``.
          It differs on every run because ``temperature`` adds randomness.
"""

from llama_course.llm import get_llm


def main() -> None:
    llm = get_llm()
    response = llm(
        "What is the most used database for data storage?",
        max_tokens=128,    # stop after at most 128 generated tokens
        temperature=0.7,   # 0 = always pick the likeliest token; higher = more varied
        top_p=0.95,        # sample only from the tokens that make up 95% of the probability
        stop=["<|eot_id|>"],
    )
    print(response["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
