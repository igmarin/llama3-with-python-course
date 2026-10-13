# Change generation settings

Control how random the answers are, how long they can be, and how much the model can read. The
ideas behind these settings are in [lesson 4](../../lessons/04_generation/).

## Steps

1. Pass settings on the call. They apply to that call only:

   ```python
   from llama_course.conversation import message_text
   from llama_course.llm import get_llm

   llm = get_llm()
   messages = [{"role": "user", "content": "Write a six-word slogan for a bakery."}]
   result = llm.create_chat_completion(
       messages=messages,
       temperature=0.0,   # 0 = repeatable; higher = more varied
       seed=7,            # fixes the random choices when temperature > 0
       max_tokens=20,     # stop after this many tokens
   )
   print(message_text(result))
   ```

2. Pick the values for the job:

   | Job | Setting |
   |-----|---------|
   | Classify, extract, call tools | `temperature=0.0` |
   | Creative writing | `temperature=0.7` to `1.2`, optionally `top_p=0.9` (picks only from the likeliest tokens) |
   | Repeatable creative output | any temperature plus a fixed `seed` |
   | Short answers | a small `max_tokens`, but not so small that JSON is cut off |

   Measured on the 3B model with that slogan prompt: three runs at `temperature=0.0` gave the same
   line three times. Three runs at `temperature=1.2` gave three different lines. Two runs at
   `temperature=1.2` with `seed=7` gave the same line twice.

3. Change the context window (how many tokens the model can read) for one run:

   ```python
   llm = get_llm(n_ctx=1024)
   print(llm.n_ctx())   # 1024
   ```

   To change it for every run of a profile, edit `n_ctx` in [`configs/models.yaml`](../../configs/models.yaml). A larger window uses
   more memory.

4. Keep a long chat inside the window. Use `trim_history` from `llama_course.context`; see
   [lesson 4](../../lessons/04_generation/).

## Notes

- `get_llm` reuses one loaded model per set of settings. Calling `get_llm(n_ctx=1024)` and
  `get_llm()` loads two models.
- Settings that are not listed here (`top_k`, `repeat_penalty`, `stop`) are also arguments of
  `create_chat_completion`. See the
  [llama-cpp-python documentation](https://llama-cpp-python.readthedocs.io/en/latest/).
