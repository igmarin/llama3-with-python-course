# Models and benchmarks

The course runs on any Llama model in GGUF format. Which one is set by a profile in
`configs/models.yaml`, chosen with `LLAMA_PROFILE`. This page records how fast each profile runs.

## Measure your machine

```bash
python scripts/download_model.py --profile laptop-16gb
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
| `laptop-16gb` | Llama 3.2 3B Q4_K_M (2.0 GB file) | MacBook Pro, Apple Silicon, 16 GB | 0.5 s | 0.02 s | 53.9 tokens/s | 2,577 MB |
| `studio-128gb` | Llama 3.3 70B Q4_K_M (about 43 GB file) | Mac Studio, 128 GB | _not measured yet_ | _not measured yet_ | _not measured yet_ | _not measured yet_ |

To fill in the second row on a bigger machine:

```bash
python scripts/download_model.py --profile studio-128gb
python scripts/benchmark.py --profile studio-128gb
```

The load time of 0.5 s was measured with the file already in the operating system's cache.
A cold start takes longer.
