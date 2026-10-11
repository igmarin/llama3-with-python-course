"""Lesson 1.2: Chat messages.

Goal:     Send a conversation instead of raw text.
Concept:  A chat is a list of messages, each with a ``role`` ("system", "user" or
          "assistant") and ``content``. ``create_chat_completion`` wraps them in the
          model's chat template for you (lesson 1.4 shows that template).
Run:      python lessons/01_basics/02_conversation.py
Expected: Four short troubleshooting steps for an internet connection.
"""

from llama_course.conversation import message_text
from llama_course.llm import get_llm


def main() -> None:
    llm = get_llm()
    messages = [
        {"role": "user", "content": "Give me four short steps to troubleshoot my internet connection."}
    ]
    result = llm.create_chat_completion(messages=messages, max_tokens=200)
    print(message_text(result))


if __name__ == "__main__":
    main()
