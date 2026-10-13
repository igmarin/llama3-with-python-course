# References

Every link here was checked and resolved when this page was written. Lesson READMEs list the same
links next to the idea they support. `python scripts/check_links.py` checks them again.

## Running Llama locally

- [llama.cpp](https://github.com/ggml-org/llama.cpp): the engine that runs the model.
- [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/): the Python
  API used in every lesson.
- [The GGUF file format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): what is inside
  a model file.
- [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md):
  the quantization tool and the types it produces (Q4_K_M and others).
- [llama.cpp on Apple Silicon: speed numbers](https://github.com/ggml-org/llama.cpp/discussions/4167):
  tokens per second reported for many Apple chips.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783): the Llama 3 paper.

## Chat and prompting

- [Llama 3 prompt format](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/):
  the special tokens and where each goes.
- [Chat templates in Hugging Face Transformers](https://huggingface.co/docs/transformers/chat_templating):
  why every model family has its own template.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165): the paper that
  showed models learn tasks from examples in the prompt.
- [Prompt Engineering Guide](https://www.promptingguide.ai/): a catalogue of prompting techniques.
- [Claude prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview):
  practical advice that applies to most chat models.

## Structured output and tools

- [Understanding JSON Schema](https://json-schema.org/understanding-json-schema/): the language
  used to fix keys and types.
- [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md): how
  output is restricted to a format.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) and
  [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/): validation, and
  schemas from Python classes.
- [Tool use in chat templates](https://huggingface.co/docs/transformers/chat_extras): how models
  are given tool descriptions.
- [Toolformer](https://arxiv.org/abs/2302.04761) and [ReAct](https://arxiv.org/abs/2210.03629):
  the research behind tool use and agents.

## Generation settings and context

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751): top-p (nucleus)
  sampling.
- [Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies):
  greedy, sampling and beam search, compared.
- [Tokenizers in the Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter2/4):
  how text becomes tokens. The [full course](https://huggingface.co/learn/llm-course) is a good
  next step.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172): models use long contexts unevenly.

## Retrieval and safety

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401):
  the paper that named RAG.
- [nomic-embed-text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) and its paper,
  [Nomic Embed](https://arxiv.org/abs/2402.01613): the embedding model used in [lesson 5](../lessons/05_rag/).
- [OWASP LLM01: Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/): the
  risk and defences.
- [Simon Willison on prompt injection](https://simonwillison.net/series/prompt-injection/): a running
  explanation of why it is hard to prevent.
- [Indirect Prompt Injection](https://arxiv.org/abs/2302.12173): attacks through retrieved
  documents.

## Fine-tuning and evaluation

- [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314): the
  fine-tuning methods behind [lesson 7](../lessons/07_finetuning/).
- [mlx-lm LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) and the
  [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html): the Apple Silicon
  tools the course uses.
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index): the same idea on PyTorch, the usual
  route on Linux and NVIDIA GPUs.
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) and
  the [reference implementation](https://github.com/google-research/google-research/tree/master/rouge):
  the scores and the code.
- [scikit-learn: model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html):
  accuracy, precision, recall and confusion matrices.
