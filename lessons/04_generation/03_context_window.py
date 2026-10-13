"""Lesson 4.3: The context window.

Goal:     Count tokens and keep a long chat from overflowing the model.
Concept:  The model can only read ``n_ctx`` tokens at once: the whole chat so far
          plus the reply. Tokens are word pieces, not words (about 1.3 per English
          word). A chat that keeps growing will exceed the window, and the model
          raises an error. ``trim_history`` drops the oldest turns and keeps the
          system prompt and the newest message.
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/04_generation/03_context_window.py
Expected: A token count, the error from an oversized prompt, then a chat of 40
          turns trimmed to fit a small budget while still answering.
"""

from llama_course.context import count_tokens, messages_tokens, trim_history
from llama_course.conversation import message_text
from llama_course.llm import get_llm

N_CTX = 512
REPLY_TOKENS = 60


def main() -> None:
    llm = get_llm(n_ctx=N_CTX)
    text = "The quick brown fox jumps over the lazy dog."
    print(f"{len(text.split())} words = {count_tokens(llm, text)} tokens")

    print(f"\nA prompt longer than n_ctx={N_CTX}:")
    try:
        llm(text * 200, max_tokens=5)
    except ValueError as exc:
        print("  error:", exc)

    system = {"role": "system", "content": "You are a concise assistant. Answer in one sentence."}
    chat = [system]
    for i in range(20):
        chat.append({"role": "user", "content": f"Fact number {i}: {text} Please remember it."})
        chat.append({"role": "assistant", "content": f"Noted fact {i}."})
    chat.append({"role": "user", "content": "What is the capital of France?"})
    print(f"\nFull chat: {len(chat)} messages, about {messages_tokens(llm, chat)} tokens")

    trimmed = trim_history(llm, chat, max_tokens=N_CTX - REPLY_TOKENS)
    print(f"Trimmed:   {len(trimmed)} messages, about {messages_tokens(llm, trimmed)} tokens")
    print("Reply:", message_text(llm.create_chat_completion(messages=trimmed, max_tokens=REPLY_TOKENS)))


if __name__ == "__main__":
    main()
