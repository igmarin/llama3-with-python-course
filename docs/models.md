# Models and benchmarks

The course runs on any Llama model in GGUF format. A **profile** in `configs/models.yaml` names one
model and its settings. Choose a profile with `LLAMA_PROFILE`, or point at any `.gguf` file with
`LLAMA_MODEL_PATH`. No lesson code changes.

## Profiles

| Profile | Model | File size | Machine it suits |
|---------|-------|-----------|------------------|
| `laptop-16gb` (default) | Llama 3.2 3B Instruct, Q4_K_M | 2.0 GB | 16 GB laptop |
| `studio-128gb` | Llama 3.3 70B Instruct, Q4_K_M | about 43 GB | 64 GB or more |
| `embedding` | nomic-embed-text v1.5, Q8_0 | 0.15 GB | any; used by the RAG lesson |

```bash
python scripts/download_model.py --profile studio-128gb
LLAMA_PROFILE=studio-128gb python lessons/04_project/classify_emails.py
```

The `studio-128gb` profile has not been run yet. The first measurement belongs in the table at the
bottom of this page.

## How much memory do I need?

A model file is roughly `parameters x bits per weight / 8`. A 3B model at 4.5 bits is about 2 GB;
a 70B model is about 43 GB. Add a few GB for the context cache and your other programs. The model
must fit in RAM, and on a Mac the GPU can use only part of it (roughly two thirds to three
quarters), so leave headroom.

**Quantization** stores each weight in fewer bits. The suffix tells you how many:

| Name | Bits per weight | Effect |
|------|-----------------|--------|
| Q4_K_M | about 4.5 | Default choice. Small, quality close to the original |
| Q5_K_M | about 5.5 | A little larger and a little better |
| Q8_0 | 8 | Nearly lossless, almost twice the size of Q4 |

Smaller files also run faster, because generating a token means reading the whole model from
memory.

## Add a profile

Copy a block in `configs/models.yaml` and change the fields:

```yaml
profiles:
  laptop-8b:
    repo: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF   # Hugging Face repo with the GGUF files
    file: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf      # one file from that repo
    n_ctx: 4096            # context window, in tokens
    n_gpu_layers: -1       # -1 = run every layer on the GPU (Metal on a Mac)
    n_threads: null        # null = let llama.cpp choose
    mlx_repo: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit   # optional: for fine-tuning
```

Then run `python scripts/download_model.py --profile laptop-8b`. The 8B file is 4.9 GB, which fits
on a 16 GB machine. This profile is an example and has not been tested here.

Use a single `.gguf` file. Large models are sometimes split into several parts, and the download
script takes one file name.

## Measure your machine

```bash
python scripts/benchmark.py --profile laptop-16gb
```

It prints a table and saves `benchmarks/<profile>.json`. Three numbers matter:

| Measure | Meaning |
|---------|---------|
| Load time | Seconds until the model is ready. The first run after a restart reads the file from disk and is slower. |
| Time to first token | How long the model reads your prompt before it starts answering. It grows with prompt length. |
| Generation speed | Tokens per second once it is answering. A token is a word piece; about 1.3 per English word. |

Peak memory is for the Python process only. On Apple Silicon the GPU shares memory with the CPU,
so treat the number as a lower bound.

## Results

Short prompts, 128 new tokens, temperature 0, median of 9 runs after one warm-up.

| Profile | Model | Machine | Load | First token | Speed | Peak memory |
|---------|-------|---------|------|-------------|-------|-------------|
| `laptop-16gb` | Llama 3.2 3B Q4_K_M | MacBook Pro, Apple Silicon, 16 GB | 0.5 s | 0.02 s | 53.9 tokens/s | 2,577 MB |
| `studio-128gb` | Llama 3.3 70B Q4_K_M | Mac Studio, 128 GB | _not measured yet_ | _not measured yet_ | _not measured yet_ | _not measured yet_ |

The load time of 0.5 s was measured with the file already in the operating system's cache. A cold
start takes longer.
