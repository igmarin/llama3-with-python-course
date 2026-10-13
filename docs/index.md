# Course documentation

Start here. To install and run your first lesson, follow the [quick start in the README](../README.md#quick-start).

## Learn

| Page | Use it to |
|------|-----------|
| [Tutorial: build a customer-message classifier](tutorial.md) | Go from zero to a working classifier in about 20 minutes |
| [Lesson pages](#lesson-pages) | Understand the idea behind each lesson, with exercises |
| [References](references.md) | Read more, grouped by topic |

## How to

| Guide | Use it to |
|-------|-----------|
| [Change models](howto/switch-models.md) | Download and switch between 3B, 8B and 70B models |
| [Add a model profile](howto/add-a-model-profile.md) | Make a new model selectable by name |
| [Run the tests](howto/run-the-tests.md) | Check your setup, with or without a model |
| [Change generation settings](howto/change-generation-settings.md) | Control randomness, answer length and context size |
| [Add your own messages](howto/add-your-own-messages.md) | Score the classifier on your own data |

## Lesson pages

Each page explains the ideas behind a lesson, walks through the code, lists common mistakes, gives
an exercise with an answer, and links to further reading. Do lessons 1 and 2 first, in order. The
others are independent. Do lesson 4 before lesson 3, because it provides the baseline score.

| Page | Covers |
|------|--------|
| [1. Basics](lessons/01-basics.md) | GGUF files, completion vs chat, system prompts, the chat template |
| [2. Prompting](lessons/02-prompting.md) | Prompt formats, few-shot, JSON mode, JSON Schema, conversation memory |
| [3. Fine-tuning](lessons/03-finetuning.md) | LoRA, train/validation/test data, accuracy, ROUGE |
| [4. Project](lessons/04-project.md) | The message classifier, honest evaluation, what examples cost |
| [5. Generation](lessons/05-generation.md) | Temperature and seed, streaming, tokens and the context window |
| [6. RAG](lessons/06-rag.md) | Embeddings, cosine search, citations, refusing, prompt injection |
| [7. Tools](lessons/07-tools.md) | Tool calling, Pydantic validation, retries |

Every lesson script also opens with a docstring: the goal, the idea, the command to run, and what
you should see.

## Reference

| Page | Read it when |
|------|--------------|
| [Models and benchmarks](models.md) | You want to know what your machine can run, and how fast |
| [Datasets](../data/README.md) | You want to know what the data contains |

## Code map

The lessons import their helpers from `src/llama_course/`:

| Module | What it does |
|--------|--------------|
| `llm.py` | `get_llm()`: loads the model chosen by the profile, once |
| `conversation.py`, `prompts.py` | Multi-turn chat and prompt builders |
| `structured.py` | JSON output, with optional Pydantic validation |
| `generation.py`, `context.py` | Streaming, token counting, trimming a long chat |
| `classify.py`, `data.py` | The message classifier and the datasets |
| `rag.py` | Chunking, search, cited answers |
| `tools.py`, `shop_tools.py` | Tool calling and the example tools |
| `finetune.py`, `rouge.py` | Training data and scoring |
| `benchmark.py` | Speed measurement |
