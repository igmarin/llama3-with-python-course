# Learn Python with local Llama models

A hands-on course. You write Python that talks to Llama models running on your own computer:
chat, prompts, structured output, answering from your documents, calling functions, and
fine-tuning. Each lesson is a short script you run and read. There are no API keys and no cloud
costs.

You need:

- Python 3.10 to 3.14. Python 3.15 is too new: the MLX package has no build for it yet.
- About 6 GB of free disk space for the models.
- 16 GB of RAM. A Mac with Apple Silicon is best. Other lessons should also run on Linux (not tested here), but
  fine-tuning (lessons 3.2 and 3.3) needs Apple Silicon.

## Quick start

```bash
git clone https://github.com/igmarin/llama3-with-python-course.git
cd llama3-with-python-course
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,finetune]"
python scripts/download_model.py --profile laptop-16gb
python lessons/01_basics/01_chat.py
```

Installing takes a few minutes, because `llama-cpp-python` compiles on your machine. The model
download is 2 GB. If the last command prints text about databases, you are ready.

## The lessons

New here? Follow the [tutorial](docs/tutorial.md): from zero to a working email classifier in about 20 minutes.

Run lessons from the repository root. Do 1 and 2 first, in order. The rest build on them but are
independent of each other.

| Lesson | You learn | Run time |
|--------|-----------|----------|
| [1. Basics](lessons/01_basics) | Load a model, ask a question, chat messages, system prompts, the chat template | seconds each |
| [2. Prompting](lessons/02_prompting) | Prompt formats, few-shot examples, JSON output, JSON Schema, a chat that remembers | seconds each |
| [3. Fine-tuning](lessons/03_finetuning) | Prepare data, train a LoRA adapter, test base against tuned, score summaries with ROUGE | 1 to 25 s, training about 1 min |
| [4. Project](lessons/04_project) | An email classifier scored on a held-out test set | 10 s |
| [5. Generation](lessons/05_generation) | Temperature and seed, streaming, the context window | seconds each |
| [6. RAG](lessons/06_rag) | Answer from your own documents, refuse when they don't say, resist prompt injection | 5 s |
| [7. Tools](lessons/07_tools) | Let the model call Python functions, validated output with Pydantic | 10 s to 1 min |

Each lesson has a page in [docs/lessons](docs/lessons) with the ideas, common mistakes, an exercise
and further reading. Times are on a MacBook Pro (Apple Silicon, 16 GB) with the 3B model. Lesson 6 needs a second small
model first: `python scripts/download_model.py --profile embedding`. Lesson 3.2 downloads a 1.8 GB
copy of the model in MLX format the first time it runs.

## Use a different model

Models are described in `configs/models.yaml`. Pick one by name, with no code changes:

```bash
python scripts/download_model.py --profile studio-128gb   # Llama 3.3 70B, a 43 GB file
LLAMA_PROFILE=studio-128gb python lessons/04_project/classify_emails.py
```

See [docs/models.md](docs/models.md) for the profiles, how to add your own, and speed
measurements.

## Check your setup

```bash
pytest -m "not model"   # about 3 seconds, needs no model
pytest                  # also runs tests against the real model
ruff check .
```

## What is where

| Folder | Contents |
|--------|----------|
| `lessons/` | The course scripts, numbered in order |
| `src/llama_course/` | Code the lessons share: model loader, chat, prompts, RAG, tools, metrics |
| `configs/models.yaml` | Model profiles |
| `data/` | Datasets, all synthetic. See [data/README.md](data/README.md) |
| `scripts/` | Model download, benchmark, data split |
| `tests/` | Tests. Most use a fake model, so they run anywhere |
| `docs/` | Tutorial, how-to guides, lesson pages. Start at [docs/index.md](docs/index.md) |
