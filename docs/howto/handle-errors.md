# Handle errors in production

The model might return invalid JSON, timeout, or run out of memory. This guide
shows how to handle these failures without crashing your application.

## What can go wrong

| Failure | Cause | Detection |
|---------|-------|-----------|
| Invalid JSON | `max_tokens` too small, or the model made a mistake | `InvalidJSONReply` from `chat_json` |
| Context too long | Chat history exceeds `n_ctx` | `ContextTooLong` from `trim_history` |
| Model not found | Model file not downloaded | `ModelNotFoundError` from `get_llm` |
| Tool call failed | Model named an unknown tool or sent bad arguments | `ToolCallFailed` from `ask` |
| Timeout | Model takes too long (large prompt, slow machine) | No built-in timeout; add one with `signal` or `threading` |
| Out of memory | Model too large for available RAM | Process killed by OS; check `dmesg` or Console |

## Retry with backoff

When a model call fails, retrying immediately often fails again. Add a delay
that grows with each attempt:

```python
import time
from llama_course.structured import InvalidJSONReply, chat_json

def chat_json_with_retry(llm, messages, schema=None, max_retries=3, **kwargs):
    """Retry a JSON chat call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return chat_json(llm, messages, schema=schema, **kwargs)
        except InvalidJSONReply as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"Retry {attempt + 1}/{max_retries} after {wait}s: {exc}")
            time.sleep(wait)
```

This works for transient failures (the model made a mistake). For persistent
failures (the prompt is too long), retrying won't help. Check the error message
first.

## Validate before you use

The model might return valid JSON that's semantically wrong: a negative price,
a part name that doesn't exist, a category that's not in your list. Validate
the output before passing it to the rest of your application:

```python
from pydantic import BaseModel, Field, ValidationError

class Product(BaseModel):
    name: str
    price_usd: float = Field(gt=0)  # must be positive
    weight_kg: float = Field(ge=0)

try:
    product = Product.model_validate(extracted_dict)
except ValidationError as exc:
    print(f"Invalid product: {exc}")
    # Fall back to a default, or ask the model to correct it
```

Lesson 6 shows how to send the error back to the model for one more try.

## Timeout long-running calls

The model can hang on a very long prompt or when the machine is under load. Add
a timeout to prevent your application from blocking forever:

```python
import signal

class ModelTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise ModelTimeout("Model call timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 seconds

try:
    result = llm.create_chat_completion(messages=messages, max_tokens=200)
except ModelTimeout:
    print("Model took too long; falling back to a default response")
    result = None
finally:
    signal.alarm(0)  # cancel the alarm
```

This works on Unix (Mac, Linux). On Windows, use `threading.Timer` instead.

## Log failures for debugging

When a model call fails, log the prompt, the error, and the settings. This
helps you spot patterns (e.g., "JSON fails when max_tokens < 50") and improve
the prompt:

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = chat_json(llm, messages, schema=schema, max_tokens=50)
except InvalidJSONReply as exc:
    logger.error(
        "JSON parse failed: max_tokens=%d, messages=%r, error=%s",
        50, messages, exc
    )
    raise
```

## Graceful degradation

When the model fails, your application should still do something useful. Options:

- **Fall back to a default response**: "I couldn't process that request. Please
  try again or contact support."
- **Return a cached response**: if you've seen this question before, use the
  cached answer (see [Cache responses](cache-responses.md)).
- **Queue for later**: save the request and retry when the model is available.
- **Route to a human**: for critical tasks, escalate to a person.

The classifier in lesson 3 uses a fallback label when the model's reply is
unusable. This keeps the application running, but the fallback might be wrong.
Logging the failure (as shown above) makes it visible.

## Where next

- [Lesson 6: Tools](../../lessons/06_tools/): validation and retries for tool calls.
- [Cache responses](cache-responses.md): avoid repeating failed calls.
- [Chain prompts](chain-prompts.md): handle errors at each step of a pipeline.