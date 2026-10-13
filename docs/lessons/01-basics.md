# Lesson 1: Basics

You load a Llama model, ask it a question, and learn the difference between plain text
completion and a chat.

**Why it matters:** Every later lesson loads a model and chats with it. This lesson shows what happens underneath, so later errors and odd answers make sense.

```bash
python lessons/01_basics/01_chat.py
python lessons/01_basics/02_conversation.py
python lessons/01_basics/03_system_prompt.py
python lessons/01_basics/04_chat_template.py
```

## The ideas

**A model file.** The model is one file in GGUF format, the format `llama.cpp` reads. It holds the
weights (the numbers the model learned) and its settings. A name like
`Llama-3.2-3B-Instruct-Q4_K_M.gguf` says: Llama 3.2, 3 billion weights, tuned to follow
instructions (`Instruct`), compressed to about 4.5 bits per weight (`Q4_K_M`). Compressing
weights is called quantization. It makes the file small enough for a laptop at a small cost in
quality.

**Completion.** A language model continues text. `llm("Why do bicycle tires lose air?")` does not
answer; it writes what would plausibly come next. That can be an answer, or more questions, or a
rambling essay. Lesson 1.1 shows this.

**Chat.** An instruct model was trained on conversations. You send a list of messages, each with a
`role` and `content`:

| Role | Who speaks |
|------|-----------|
| `system` | You, setting the rules before the user speaks |
| `user` | The person asking |
| `assistant` | The model's earlier replies |

`create_chat_completion(messages=...)` returns the next assistant message. Lesson 1.3 uses a
system prompt to turn the model into a bike shop's help desk that refuses anything else.

**The chat template.** The model only reads text. Roles are marked with special tokens such as
`<|start_header_id|>user<|end_header_id|>`. `create_chat_completion` adds them for you. Lesson 1.4
writes them by hand with `llama3_chat_prompt` so you can see what the model receives. It also
shows why you stop generation on `<|eot_id|>` ("end of turn"): without a stop, the model keeps
writing.

## Code map

| File | What it shows |
|------|---------------|
| `src/llama_course/llm.py` | `get_llm()` loads the model chosen by the profile, once per run |
| `src/llama_course/prompts.py` | `llama3_chat_prompt` builds the raw template |
| `src/llama_course/conversation.py` | `message_text` pulls the reply out of a result |

## Common mistakes

- **Using plain completion for a question.** If the reply rambles, switch to the chat API.
- **Forgetting `max_tokens`.** Generation stops at a default limit that may cut your answer off
  mid-sentence, or run longer than you want. Set it.
- **Putting `<|begin_of_text|>` in a hand-written prompt.** llama.cpp adds it, and you get a
  warning about a duplicate token. `llama3_chat_prompt` leaves it out for this reason.
- **Expecting the same answer every run.** The default `temperature` adds randomness. Lesson 4
  covers it.

## Exercise

1. Add `temperature=0.0` to the call in `01_chat.py` and run it twice. What changes?
2. How many tokens does the prompt in lesson 1.4 use?

<details>
<summary>Answer</summary>

1. The two outputs are identical. Temperature 0 always picks the most likely next token. (Checked on the
   3B model with a 30-token limit.)
2. Add these lines after `prompt` is built:

   ```python
   print(len(llm.tokenize(prompt.encode(), add_bos=False)))
   ```

   It prints 75 for the 246-character prompt: tokens are word pieces, not words or characters.
</details>

## Go deeper

- [llama.cpp](https://github.com/ggml-org/llama.cpp): the engine that runs the model.
- [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/): the Python
  API used in every lesson.
- [The GGUF file format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): what is inside
  a model file.
- [Llama 3 prompt format](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/):
  the special tokens and where each goes.
- [Chat templates in Hugging Face Transformers](https://huggingface.co/docs/transformers/chat_templating):
  why every model family has its own template.
