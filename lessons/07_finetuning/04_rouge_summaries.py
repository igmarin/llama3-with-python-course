"""Lesson 7.4: Scoring generated text with ROUGE.

Goal:     Measure summaries, where there is no single correct answer.
Concept:  Accuracy works for labels. A summary can be right in many wordings, so we
          compare it with a human reference by word overlap (ROUGE). It is crude but
          cheap, and useful for comparing two models or two prompts on the same
          texts, such as a base model and a fine-tuned one. This lesson scores the
          base model's one-sentence summaries of data/summaries.csv.
Requires: pip install -e ".[finetune]"
Run:      python lessons/07_finetuning/04_rouge_summaries.py
Expected: One example summary next to its reference, then ROUGE-1/2/L averages over
          31 texts. Last run: ROUGE-1 0.51, ROUGE-2 0.22, ROUGE-L 0.43.
"""

from llama_course.conversation import message_text
from llama_course.data import load_summaries
from llama_course.llm import get_llm
from llama_course.rouge import rouge_scores


def main() -> None:
    llm = get_llm()
    df = load_summaries()
    predictions = []
    for text in df["text"]:
        messages = [
            {"role": "system", "content": "Summarize the text in one short sentence."},
            {"role": "user", "content": text},
        ]
        predictions.append(message_text(llm.create_chat_completion(messages=messages, max_tokens=60, temperature=0.0)).strip())

    print("Model:    ", predictions[0])
    print("Reference:", df["summary"][0], "\n")
    for name, score in rouge_scores(predictions, list(df["summary"])).items():
        print(f"{name}: {score:.3f}")


if __name__ == "__main__":
    main()
