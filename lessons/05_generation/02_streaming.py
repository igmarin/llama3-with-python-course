"""Lesson 5.2: Streaming.

Goal:     Show the reply as it is generated instead of after it finishes.
Concept:  A model produces text one token at a time. With ``stream=True`` you get
          each token as soon as it exists, so a user starts reading right away.
          Time to first token matters more for chat feel than total time.
Run:      python lessons/05_generation/02_streaming.py
Expected: A short story that appears word by word, then the time to first token
          and the total time.
"""

import time

from llama_course.generation import stream_chat
from llama_course.llm import get_llm


def main() -> None:
    llm = get_llm()
    messages = [{"role": "user", "content": "Tell a three-sentence story about a robot who learns to paint."}]

    start = time.perf_counter()
    first = None
    for piece in stream_chat(llm, messages, max_tokens=150):
        if first is None:
            first = time.perf_counter() - start
        print(piece, end="", flush=True)
    total = time.perf_counter() - start
    print(f"\n\nTime to first token: {first:.2f}s. Total: {total:.2f}s.")


if __name__ == "__main__":
    main()
