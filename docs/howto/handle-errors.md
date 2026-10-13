# Handle model errors

What can fail when you call the model, and what to do about each failure. The main rule for a local
model: retrying the same request does not help. Change the request.

## What can go wrong

| Failure | Cause | What you see |
|---------|-------|--------------|
| Invalid JSON | `max_tokens` cut the reply off, or no schema was given | `InvalidJSONReply` from `chat_json` |
| Valid JSON, wrong values | The model made a mistake or invented a value | `ValidationError` from Pydantic, or nothing at all |
| Unknown tool or bad arguments | The model named a tool that does not exist | `ToolCallFailed` from `ask` |
| Chat too long | The history no longer fits in `n_ctx` | `ContextTooLong` from `trim_history` |
| Model missing | The model file is not downloaded | `ModelNotFoundError` from `get_llm` |
| Out of memory | The model is too big for this machine | The process is killed, or the machine freezes |

## Change the request, do not repeat it

A cloud API can fail because of the network or a busy server, so waiting and retrying the same
call makes sense there. A local model at temperature 0 gives the same reply to the same prompt. A
request that failed once fails again:

```python
from llama_course.llm import get_llm
from llama_course.structured import InvalidJSONReply, chat_json

llm = get_llm()
msgs = [{"role": "user", "content": "Describe the Ridgeline 29 bike as JSON with name, price_usd and colours."}]
for max_tokens in (10, 10, 10, 200):
    try:
        print(max_tokens, "ok", chat_json(llm, msgs, max_tokens=max_tokens, temperature=0.0))
    except InvalidJSONReply as exc:
        print(max_tokens, "fail", exc)
```

On the 3B model the first three calls failed in exactly the same place
(`'{"name": "Ridgeline 29'`), and the call with `max_tokens=200` worked. Fix the cause instead:

- Cut-off JSON: raise `max_tokens`.
- Wrong shape: pass a schema, so generation is constrained to it.
- Wrong values: send the error back to the model and let it try again. `chat_model` in
  [`structured.py`](../../src/llama_course/structured.py) and `ask` in
  [`tools.py`](../../src/llama_course/tools.py) both do this once by default.

## Validate values, not only the format

The successful call above returned `'price_usd': 1200`. The prompt contains no price, so the model
invented one. The JSON is valid and would pass a type check. Two defences:

- Check what you can with Pydantic: ranges, allowed values, required fields. `chat_model` sends
  these errors back to the model.
- Do not ask the model for facts it was never given. Look prices up in your own data, or with a
  tool ([lesson 6](../../lessons/06_tools/)).

## Limit how long a call can take

`max_tokens` is the practical time limit: generation stops after that many tokens. At the 3B
model's 45 tokens per second, `max_tokens=200` is at most about 5 seconds of generation, plus the
time to read the prompt.

A hard timeout that stops a call halfway is harder. Python cannot safely interrupt the model from
another thread. If you need one, run the model in a separate process and stop that process.

## Fall back, and make the fallback visible

When the model still fails, the program needs a safe default. The classifier in
[lesson 3](../../lessons/03_project/) uses the label `Feedback`. `classify` logs a warning when it
does, so you can count how often it happens:

```python
import logging

logging.basicConfig(level=logging.WARNING)
```

Other safe defaults: send the message to a person, or tell the user to try again. Logs contain
customer messages, so keep them where customer data is allowed to be.

## Where next

- [Lesson 6: Tools](../../lessons/06_tools/): sending validation errors back to the model.
- [Change generation settings](change-generation-settings.md): `max_tokens`, temperature and context size.
