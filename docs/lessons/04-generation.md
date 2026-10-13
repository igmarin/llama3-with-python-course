# Lesson 4: Generation

You control how the model writes: how random it is, when you see the text, and how much it can
read.

**Why it matters:** The same prompt can give different answers, and a long chat eventually stops fitting. This lesson gives you control over both.

```bash
python lessons/04_generation/01_sampling.py
python lessons/04_generation/02_streaming.py
python lessons/04_generation/03_context_window.py
```

## The ideas

### Sampling

At each step the model gives every possible next token a probability, then one is
picked. These settings change the pick:

| Setting | Effect |
|---------|--------|
| `temperature` | 0 always takes the likeliest token, so output is repeatable. Higher values take unlikely tokens more often |
| `top_p` | Pick only from the smallest set of tokens whose probabilities add up to `top_p`. This cuts the unlikely tail |
| `top_k` | Pick only from the `k` likeliest tokens |
| `seed` | Fixes the random choices, so the same settings give the same text |

For classification and extraction use temperature 0. For writing, 0.7 is a common start.

### Streaming

The model writes one token at a time. With `stream=True` you receive each token as it
exists, so the reader starts early. Time to first token matters more for chat than total time.
`stream_chat` yields the text pieces.

### Tokens and the context window

A token is a word piece, about 1.3 per English word. The model
reads at most `n_ctx` tokens at once, and that budget covers your whole chat plus the reply. A
longer prompt raises `ValueError: Requested tokens (1802) exceed context window of 512`.
`trim_history` drops the oldest turns until the chat fits. It keeps the system prompt and the
newest message, and it never leaves the history starting with an assistant reply.

How `trim_history` keeps a long chat inside the context window:

```mermaid
flowchart TD
    H[Chat history + new message] --> F{"Fits in n_ctx<br/>minus room for the reply?"}
    F -- yes --> Send[Send to the model]
    F -- no --> K{"Do system prompt +<br/>newest message fit?"}
    K -- no --> Err[ContextTooLong error]
    K -- yes --> Drop[Drop the oldest turn]
    Drop --> F
```

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/generation.py` | `stream_chat` |
| `src/llama_course/context.py` | `count_tokens`, `messages_tokens`, `trim_history`, `ContextTooLong` |
| `get_llm(n_ctx=512)` | Overrides a profile setting for one call |

## Common mistakes

- Expecting the same output at temperature 0.7: pass a `seed` if you need repeatability.
- Forgetting the reply needs room: trim to `n_ctx` minus the reply length, not to `n_ctx`.
- Counting words instead of tokens: the token count is higher, and it differs per model.
- Per-message overhead: each chat message costs a few extra tokens for role markers. The code
  adds 8 per message, an estimate for Llama 3.
- Raising `n_ctx` without cost: a larger window uses more memory.

## Exercise

What does `trim_history` do when the system prompt alone is longer than the budget?

<details>
<summary>Answer</summary>

It raises `ContextTooLong`: "System prompt and newest message need more than 512 tokens." Trimming
old turns cannot help, because the system prompt and the newest message are always kept. You must
shorten the prompt or raise `n_ctx`. Try it with a 600-word system prompt and `max_tokens=512`.
</details>

## Go deeper

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751): the paper that
  introduced top-p (nucleus) sampling.
- [Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies):
  greedy, sampling and beam search, compared.
- [Tokenizers in the Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter2/4):
  how text becomes tokens.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172): models use long contexts unevenly. More
  context is not always better.
- [llama.cpp on Apple Silicon: speed numbers](https://github.com/ggml-org/llama.cpp/discussions/4167):
  tokens per second reported for many Apple chips.
