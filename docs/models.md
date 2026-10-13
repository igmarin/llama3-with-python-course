# Models and benchmarks

The course runs on any Llama model in GGUF format (see [lesson 1](../lessons/01_basics/)). A
profile is a named model setup: one entry in [`configs/models.yaml`](../configs/models.yaml) with
the model file and its settings. Choose a profile with the `LLAMA_PROFILE` environment variable, or
point at any `.gguf` file with `LLAMA_MODEL_PATH`. No lesson code changes.

New words are in the [glossary](glossary.md).

## Profiles

| Profile | Model | File size | Machine it suits |
|---------|-------|-----------|------------------|
| `laptop-16gb` (default) | Llama 3.2 3B Instruct, Q4_K_M | 2.0 GB | 16 GB laptop |
| `laptop-16gb-q8` | Llama 3.2 3B Instruct, Q8_0 | 3.4 GB | 16 GB laptop; compared in [lesson 8](../lessons/08_quantization/) |
| `laptop-8b` | Llama 3.1 8B Instruct, Q4_K_M | 4.9 GB | 16 GB laptop |
| `studio-128gb` | Llama 3.3 70B Instruct, Q4_K_M | about 43 GB | 64 GB or more |
| `embedding` | nomic-embed-text v1.5, Q8_0 | 0.15 GB | any; used by the RAG lesson |

```bash
python scripts/download_model.py --profile studio-128gb
LLAMA_PROFILE=studio-128gb python lessons/03_project/classify_messages.py
```

The `studio-128gb` profile has not been run yet. The first measurement belongs in the table at the
bottom of this page.

## How much memory do I need?

A model file's size in bytes is roughly `number of weights x bits per weight / 8`. A 3B model at
4.5 bits per weight is about 2 GB, and a 70B model is about 43 GB.

The whole model must fit in memory (RAM), with room to spare. Add a few GB for the context cache,
the memory the model uses to hold your conversation while it works, and for your other programs.
On a Mac the GPU, the chip that does the model's arithmetic, can use only part of the RAM: roughly
two thirds to three quarters. So a 16 GB Mac is comfortable with the 2 GB and 4.9 GB models, and a
43 GB model needs a much bigger machine.

Quantization stores each weight in fewer bits. The suffix tells you how many:

| Name | Bits per weight | Effect |
|------|-----------------|--------|
| Q4_K_M | about 4.5 | Default choice. Small, quality close to the original |
| Q5_K_M | about 5.5 | A little larger and a little better |
| Q8_0 | 8 | Very close to the original quality, almost twice the size of Q4 |

Smaller files also run faster, because generating a token means reading the whole model from
memory.

## Add a profile

Copy a block in [`configs/models.yaml`](../configs/models.yaml), change the fields, and download it. The step-by-step guide is
[Add a model profile](howto/add-a-model-profile.md). Use a single `.gguf` file; a model split into
several parts will not work with the download script.

## Measure your machine

```bash
python scripts/benchmark.py --profile laptop-16gb
```

It prints a table and saves [`benchmarks/<profile>.json`](../benchmarks/). Three numbers matter:

| Measure | Meaning |
|---------|---------|
| Load time | Seconds until the model is ready. The first run after a restart reads the file from disk and is slower. |
| Time to first token | How long the model reads your prompt before it starts answering. It grows with prompt length. |
| Generation speed | Tokens per second once it is answering. A token is a word piece; about 1.3 per English word. |

Peak memory is for the Python process only. On Apple Silicon the GPU shares memory with the CPU,
so treat the number as a lower bound.

## Results

Short prompts, 128 new tokens, temperature 0. Each number is the median (the middle value) of 9
runs, after one untimed warm-up run.

| Profile | Model | Machine | Load | First token | Speed | Peak memory |
|---------|-------|---------|------|-------------|-------|-------------|
| `laptop-16gb` | Llama 3.2 3B Q4_K_M | MacBook Pro, Apple Silicon, 16 GB | 0.5 s | 0.02 s | 45.4 tokens/s | 2,577 MB |
| `laptop-8b` | Llama 3.1 8B Q4_K_M | MacBook Pro, Apple Silicon, 16 GB | 2.7 s | 0.04 s | 23.2 tokens/s | 5,252 MB |
| `studio-128gb` | Llama 3.3 70B Q4_K_M | Mac Studio, 128 GB | _not measured yet_ | _not measured yet_ | _not measured yet_ | _not measured yet_ |

Generation speed drops as the model grows: the 8B model has almost three times as many weights as the
3B model and runs at about half the speed. Speeds also vary between runs: an earlier run of the 3B
model with slightly different prompts measured 53.9 tokens/s. Compare models on the same day and
prompts.

The bigger model is more accurate on the hard cases. On the message classifier in [lesson 3](../lessons/03_project/), both
models pass all 33 test messages and score 98.3% on the 60-message challenge set, but on the 27
hard training messages the 8B model scored 96.3% against 81.5% for the 3B model.

The load time of 0.5 s was measured with the file already in the operating system's cache. The
first load after a restart reads the file from disk and takes longer.
