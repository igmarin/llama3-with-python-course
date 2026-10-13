"""Lesson 1.4: The chat template.

Goal:     See what the model really receives when you send chat messages.
Concept:  Llama 3 was trained on text with special tokens that mark who is
          speaking (``<|start_header_id|>user<|end_header_id|>`` and so on).
          ``create_chat_completion`` adds them for you. Here we write them by
          hand with ``llama3_chat_prompt`` and call plain completion.
          (llama.cpp adds the start-of-text token itself, so we leave it out.)
          Stop on ``<|eot_id|>`` ("end of turn"), or the model keeps going.
Run:      python lessons/01_basics/04_chat_template.py
Expected: The prompt with its special tokens, then a one-sentence answer saying a
          tandem bicycle seats two riders.
"""

from llama_course.llm import get_llm
from llama_course.prompts import llama3_chat_prompt


def main() -> None:
    llm = get_llm()
    prompt = llama3_chat_prompt(
        "You are a helpful assistant. Answer in one sentence.",
        "How many riders does a tandem bicycle carry?",
    )
    print(prompt)
    response = llm(prompt, max_tokens=128, temperature=0.7, top_p=0.95, stop=["<|eot_id|>"])
    print(response["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
