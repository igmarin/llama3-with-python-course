# Learn Python with local Llama models

A hands-on course. You write Python that talks to Llama models running on your own computer:
chat, prompts, structured output, answering from your documents, calling functions, and
fine-tuning. Every example is built around one made-up bike shop. There are no API keys and no
cloud costs.

Needs a Mac with Apple Silicon, 16 GB of RAM and Python 3.10 to 3.14. [Setup details](docs/setup.md)

## Start in 5 minutes

```bash
git clone https://github.com/igmarin/llama3-with-python-course.git
cd llama3-with-python-course
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,finetune]"
python scripts/download_model.py --profile laptop-16gb
python lessons/01_basics/01_chat.py
```

If the last command prints text about bicycle tires, you are ready.

## Then follow this path

1. **[Tutorial](docs/tutorial.md)**: build a classifier for the shop's customer messages, about 20
   minutes.
2. **The lessons**, in order:

   | Lesson | You learn |
   |--------|-----------|
   | [1. Basics](docs/lessons/01-basics.md) | Load a model, chat messages, system prompts, the chat template |
   | [2. Prompting](docs/lessons/02-prompting.md) | Prompt formats, few-shot examples, JSON output, a chat that remembers |
   | [3. Project](docs/lessons/03-project.md) | Sort customer messages and measure the accuracy honestly |
   | [4. Generation](docs/lessons/04-generation.md) | Temperature and seed, streaming, the context window |
   | [5. RAG](docs/lessons/05-rag.md) | Answer from your own documents, resist prompt injection |
   | [6. Tools](docs/lessons/06-tools.md) | Let the model call Python functions, validated output |
   | [7. Fine-tuning](docs/lessons/07-finetuning.md) | Train a LoRA adapter and test whether it helped |

   Each page explains the ideas and links to its scripts in `lessons/`.

3. **Everything else** is in the [docs index](docs/index.md): how-to guides (switch to a bigger
   model, add your own data, run the tests), model benchmarks, and further reading.
