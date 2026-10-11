"""Lesson 2.5: A conversation that remembers.

Goal:     Ask follow-up questions that depend on earlier answers.
Concept:  The model has no memory. ``Conversation`` keeps the message list and
          sends all of it each turn, which is why "a different city in the same
          country" works. The history grows every turn and must fit in ``n_ctx``.
Run:      python lessons/02_prompting/05_conversation_class.py
Expected: A "City, Country" answer, then a second city in the same country.
"""

from llama_course.conversation import Conversation
from llama_course.llm import get_llm

SYSTEM = (
    "You are a travel expert that recommends a travel destination based on a prompt. "
    "Return the location name only as 'City, Country'."
)


def main() -> None:
    llm = get_llm()

    # Single turn
    chatbot = Conversation(llm, system_prompt=SYSTEM)
    print(chatbot.create_completion("I'd like to learn about the Aztecs.", max_tokens=20))

    # Two turns: the second only makes sense with the first in history
    chatbot = Conversation(llm, system_prompt=SYSTEM)
    print(chatbot.create_completion("Recommend a Spanish-speaking city.", max_tokens=20))
    print(chatbot.create_completion("A different city in the same country", max_tokens=20))


if __name__ == "__main__":
    main()
