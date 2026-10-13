"""Lesson 2.5: A conversation that remembers.

Goal:     Ask follow-up questions that depend on earlier answers.
Concept:  The model has no memory. ``Conversation`` keeps the message list and
          sends all of it each turn, which is why "something lighter for the same
          ride" works: the model still sees the first request. The history grows
          every turn and must fit in ``n_ctx``.
Run:      python lessons/02_prompting/05_conversation_class.py
Expected: A bike type for each request. The follow-up stays on the same kind of
          riding.
"""

from llama_course.conversation import Conversation
from llama_course.llm import get_llm

SYSTEM = (
    "You are a bike shop assistant. Recommend one type of bicycle for the rider's "
    "needs. Reply with the bike type and one short reason."
)


def main() -> None:
    llm = get_llm()

    # Single turn
    chatbot = Conversation(llm, system_prompt=SYSTEM)
    print(chatbot.create_completion("I want to ride forest trails with roots and rocks.", max_tokens=40))

    # Two turns: the second only makes sense with the first in history
    chatbot = Conversation(llm, system_prompt=SYSTEM)
    print(chatbot.create_completion("I commute 12 km to work on city streets.", max_tokens=40))
    print(chatbot.create_completion("Something I can fold and take on the train for the same ride.", max_tokens=40))


if __name__ == "__main__":
    main()
