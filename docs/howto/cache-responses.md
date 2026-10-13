# Cache model responses

Avoid repeating the same model calls by storing responses on disk. This is
useful when the same question is asked repeatedly, or when you're iterating on
a pipeline and don't want to wait for the model each time.

## When to cache

Cache when:

- The same question is asked multiple times (e.g., "What's your return policy?").
- You're developing a pipeline and want to skip the model call while testing
  later steps.
- You want to compare two prompts on the same inputs without paying the cost
  twice.

Don't cache when:

- The answer should change over time (e.g., stock levels, prices).
- You're using temperature > 0 and want varied outputs.
- The prompt includes a timestamp or other dynamic data.

## A simple disk cache

Hash the prompt and settings, store the response in a file. Next time, check
the cache first:

```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_key(messages: list[dict], **kwargs) -> str:
    """Hash the messages and settings into a filename."""
    content = json.dumps({"messages": messages, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def cached_chat_completion(llm, messages, cache=True, **kwargs):
    """Return a cached response if available, otherwise call the model."""
    if not cache:
        return llm.create_chat_completion(messages=messages, **kwargs)
    
    key = cache_key(messages, **kwargs)
    cache_file = CACHE_DIR / f"{key}.json"
    
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    
    result = llm.create_chat_completion(messages=messages, **kwargs)
    cache_file.write_text(json.dumps(result))
    return result
```

Use it like the normal `create_chat_completion`:

```python
from llama_course.llm import get_llm

llm = get_llm()
messages = [{"role": "user", "content": "Why do bicycle tires lose air?"}]

# First call: hits the model, takes a few seconds
result1 = cached_chat_completion(llm, messages, max_tokens=100)

# Second call: reads from cache, instant
result2 = cached_chat_completion(llm, messages, max_tokens=100)

assert result1 == result2
```

## Cache invalidation

The cache key includes the messages and settings. If you change the prompt, the
cache key changes, and you get a fresh response. To clear the cache:

```bash
rm -rf cache/
```

For finer control, add a timestamp to the cache file and delete files older
than a certain age:

```python
import time

for cache_file in CACHE_DIR.glob("*.json"):
    if time.time() - cache_file.stat().st_mtime > 86400:  # 1 day
        cache_file.unlink()
```

## What to cache

Cache the full `create_chat_completion` result, not just the text. This
preserves the token counts, finish reason, and other metadata. If you only need
the text, extract it after loading from cache:

```python
result = cached_chat_completion(llm, messages, max_tokens=100)
text = result["choices"][0]["message"]["content"]
```

## Limitations

- The cache is local to your machine. It doesn't sync across devices.
- The cache doesn't know if the model has been updated. If you download a new
  model file, clear the cache.
- The cache grows unbounded. Add a cleanup script or use a cache library like
  `diskcache` for size limits.

## Where next

- [Chain prompts](chain-prompts.md): cache intermediate results in a pipeline.
- [Handle errors](handle-errors.md): retry failed calls instead of caching them.
- [Lesson 3: Project](../../lessons/03_project/): a place to try it: cache the
  classifier's replies while you change the report code.