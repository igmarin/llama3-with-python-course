# Change models

Run the course on a different Llama model without editing any code.

## Steps

1. See the models available:

   ```bash
   python -c "from llama_course.llm import load_profiles; print(list(load_profiles()[1]))"
   ```

   You get `['laptop-16gb', 'laptop-8b', 'studio-128gb', 'embedding']`. The `embedding` profile is
   not a chat model; lesson 5 uses it.

2. Download the one you want:

   ```bash
   python scripts/download_model.py --profile laptop-8b
   ```

   The 8B model is a 4.9 GB file and took about 3.5 minutes to download here.

3. Run any lesson with that profile:

   ```bash
   LLAMA_PROFILE=laptop-8b python lessons/01_basics/01_chat.py
   ```

   To keep it for the whole terminal session, run `export LLAMA_PROFILE=laptop-8b` once.

4. Check which profile is active:

   ```bash
   LLAMA_PROFILE=laptop-8b python -c "from llama_course.llm import get_profile; print(get_profile().name)"
   ```

## Use any GGUF file you already have

```bash
LLAMA_MODEL_PATH=/path/to/model.gguf python lessons/01_basics/01_chat.py
```

The file replaces the profile's file. The profile's other settings (context size, GPU layers)
still apply.

## What to expect

On a 16 GB MacBook Pro (Apple Silicon), same prompts, Q4_K_M files:

| Profile | Speed | Peak memory | Test messages (33) | Training messages (129) | Hard training messages (27) |
|---------|-------|-------------|--------------------|-------------------------|-----------------------------|
| `laptop-16gb` (3B) | 45.4 tokens/s | 2.6 GB | 100% | 95.3% | 81.5% |
| `laptop-8b` (8B) | 23.2 tokens/s | 5.3 GB | 100% | 97.7% | 96.3% |

Accuracy is for the message classifier in lesson 3 with its default prompt. Both models pass every
test message, so the training messages show the difference: the 8B model handles the hard ones much
better, at half the speed. See [Models and benchmarks](../models.md).

## If something fails

- `ModelNotFoundError: Model file not found: ...`: the file is not downloaded. The message ends
  with the command to run, for example `python scripts/download_model.py --profile studio-128gb`.
- `KeyError: "Unknown profile 'nope'. Available: ..."`: the name is misspelled. The message lists
  the valid names.
- The model loads but the machine swaps or freezes: the model is too big for your memory. Pick
  a smaller profile. [Models and benchmarks](../models.md) explains how to estimate the size.
