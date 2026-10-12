# Course documentation

Start with the [README](../README.md) to install and run your first lesson.

## Learning order

1. **Basics** (`lessons/01_basics`): load a model and chat with it.
2. **Prompting** (`lessons/02_prompting`): control the format of what you get back.
3. Then any of these, in any order:
   - **Project** (`lessons/04_project`): classify emails and measure the accuracy.
   - **Generation** (`lessons/05_generation`): sampling, streaming, context limits.
   - **RAG** (`lessons/06_rag`): answer from your own documents.
   - **Tools** (`lessons/07_tools`): let the model call functions.
   - **Fine-tuning** (`lessons/03_finetuning`): train it to do your task better. Do the project first,
     because it provides the baseline score.

Every lesson file opens with a docstring: the goal, the idea, the command to run, and what you
should see.

## Reference pages

| Page | Read it when |
|------|--------------|
| [Models and benchmarks](models.md) | You want a different model, or want to know what your machine can run |
| [Datasets](../data/README.md) | You want to know what the data contains or add your own |

## Code map

The lessons import their helpers from `src/llama_course/`:

| Module | What it does |
|--------|--------------|
| `llm.py` | `get_llm()`: loads the model chosen by the profile, once |
| `conversation.py`, `prompts.py` | Multi-turn chat and prompt builders |
| `structured.py` | JSON output, with optional Pydantic validation |
| `generation.py`, `context.py` | Streaming, token counting, trimming a long chat |
| `classify.py`, `data.py` | The email classifier and the datasets |
| `rag.py` | Chunking, search, cited answers |
| `tools.py`, `shop_tools.py` | Tool calling and the example tools |
| `finetune.py`, `rouge.py` | Training data and scoring |
| `benchmark.py` | Speed measurement |
