# Setup

Everything about your computer and the install. The [README](../README.md) has the short version.

## What you need

| Item | Requirement |
|------|-------------|
| Computer | A Mac with Apple Silicon. Lesson 7 (fine-tuning) requires one. Other lessons should also run on Linux, but this was not tested here |
| Memory | 16 GB of RAM for the default 3B model |
| Disk | About 6 GB free for the models |
| Python | 3.10 to 3.14. Python 3.15 is too new: the MLX package used in lesson 7 has no build for it yet |

Check your Python version with `python3 --version`. If it prints 3.15, create the virtual environment
with an older version, for example `python3.14 -m venv .venv`.

## Install

```bash
git clone https://github.com/igmarin/llama3-with-python-course.git
cd llama3-with-python-course
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,finetune]"
python scripts/download_model.py --profile laptop-16gb
python lessons/01_basics/01_chat.py
```

The install takes a few minutes, because `llama-cpp-python` compiles on your machine. The model
download is 2 GB. If the last command prints text about bicycle tires, you are ready.

Run every command from the repository root, with the virtual environment active
(`source .venv/bin/activate` in each new terminal).

## Extra downloads

Two lessons need more than the default model. Each downloads once.

| Lesson | Needs | How |
|--------|-------|-----|
| 5. RAG | A small embedding model (0.15 GB) | `python scripts/download_model.py --profile embedding` |
| 7. Fine-tuning | The 3B model in MLX format (1.8 GB) | Downloaded automatically the first time lesson 7.2 runs |

## Check that it works

```bash
pytest -m "not model"   # about 3 seconds, needs no model
pytest                  # also runs tests against the real model
```

More in [Run the tests](howto/run-the-tests.md).

## Run times

Measured on a MacBook Pro (Apple Silicon, 16 GB) with the 3B model.

| Lesson | Run time |
|--------|----------|
| 1. Basics, 2. Prompting | a few seconds per script |
| 3. Project | about 10 s |
| 4. Generation | a few seconds per script |
| 5. RAG | about 5 s |
| 6. Tools | 10 s; lesson 6.2 about 50 s |
| 7. Fine-tuning | 1 to 25 s per script; training about 1 minute |

## Common install problems

| Problem | Fix |
|---------|-----|
| `ResolutionImpossible` mentioning `mlx` | Your Python is 3.15. Recreate the virtual environment with Python 3.14 |
| `ModelNotFoundError: Model file not found` | Run the download command that the error message prints |
| The machine slows down or freezes when loading a model | The model is too big for your memory. See [Models and benchmarks](models.md) |
