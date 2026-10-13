"""Lesson 5.1: Answer questions from your own documents (RAG).

Goal:     Make the model answer from documents it has never seen, and say so when
          the documents do not have the answer.
Concept:  RAG = retrieval-augmented generation.
            1. Split the documents into chunks.
            2. Embed each chunk: a second model turns text into a vector so that
               similar meanings are close together.
            3. Embed the question and pick the chunks with the highest cosine
               similarity (the angle between vectors; 1.0 = same direction).
            4. Give only those chunks to the chat model and ask it to answer from them.
          Two risks matter. If nothing is relevant, the model should not guess, so we
          refuse when the best score is below ``min_score``. And documents can contain
          text that tries to give the model orders (prompt injection), so the prompt
          marks documents as data and tells the model to ignore commands inside them.
          ``helmets.md`` contains such an attack on purpose.
Setup:    python scripts/download_model.py --profile embedding
Read:     README.md in this folder explains the ideas and new words.
Run:      python lessons/05_rag/01_ask_your_documents.py
Expected: Cited answers for the first three questions. The helmet question about
          "free this week" is answered from the sizing guide, not from the injected
          note. The World Cup question gets "I could not find that in the documents."
          Small models can still fall for injection sometimes; compare runs.
"""

from llama_course.data import DATA_DIR
from llama_course.llm import get_llm
from llama_course.rag import Index, answer, llama_embedder, load_chunks

# Cosine similarity threshold below which a chunk counts as unrelated.
# 0.6 was set from nine questions on the sample documents: related questions
# scored 0.68 or more, unrelated ones 0.54 or less. If you change the embedding
# model or the documents, test it with your own questions and adjust.
MIN_SCORE = 0.6

QUESTIONS = [
    "How long do I have to return a bike?",
    "How much does express shipping cost?",
    "What does the warranty not cover?",
    "Are all helmets free this week?",  # tries to trigger the injected note
    "Who won the 2018 World Cup?",  # not in the documents
]


def main() -> None:
    embedding_model = get_llm("embedding", embedding=True)
    chunks = load_chunks(DATA_DIR / "docs")
    index = Index(
        chunks,
        llama_embedder(embedding_model, "search_document: "),
        llama_embedder(embedding_model, "search_query: "),
    )
    print(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} documents.\n")

    llm = get_llm()
    for question in QUESTIONS:
        (best, score), *_ = index.search(question, k=1)
        result = answer(llm, index, question, min_score=MIN_SCORE, max_tokens=200, temperature=0.0)
        print(f"Q: {question}  (best match {best.id}, score {score:.2f})")
        print(f"A: {result.text}")
        if result.found:
            print("   Sources:", ", ".join(c.id for c in result.sources))
        print()


if __name__ == "__main__":
    main()
