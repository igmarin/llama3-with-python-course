# Run the tests

Check that your setup works, with or without a model on disk.

## Steps

1. Run the fast tests. They use a fake model and need no download:

   ```bash
   pytest -m "not model"
   ```

   You see `109 passed, 4 deselected` in about 2 seconds.

2. Run the tests that use the real model (needs the model for the default profile and the
   `embedding` profile):

   ```bash
   pytest -m model
   ```

   You see `4 passed`. They load the model, generate a token, check that a seed repeats the same
   text, and search real embeddings.

3. Run everything, and the style check:

   ```bash
   pytest
   ruff check .
   ```

   You see `113 passed` and `All checks passed!`.

## Test a different model

```bash
LLAMA_PROFILE=laptop-8b pytest -m model
```

## If the model tests are skipped

`pytest -m model -rs` prints why. With no model file you see, for example,
`SKIPPED ... model file missing: /nonexistent`. Download the model, or fix `LLAMA_MODEL_PATH`.
`llama_cpp is not installed` means the install step did not finish.

## What the tests cover

Most tests replace the model with `FakeLLM` (in `tests/conftest.py`), which returns canned replies and
records what it was sent. That makes them fast and repeatable, and lets them check how your code
handles bad replies such as broken JSON. Add a test the same way:

```python
def test_my_helper(fake_llm):
    llm = fake_llm(["Repair"])
    ...  # call your code with llm, then check llm.calls
```
