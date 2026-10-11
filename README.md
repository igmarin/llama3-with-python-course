# Llama3 with Python

Minimal example using [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) to run a Llama 3 GGUF model locally on CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # add ".[finetune]" for lesson 3
```

Dependencies are declared in `pyproject.toml` (core, `finetune`, `dev`).

## Download a model

A small GGUF model is recommended for CPU-only use. Use the helper script:

```bash
python download_model.py --model bartowski/Llama-3.2-3B-Instruct-GGUF \
    --file Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Or download manually from HuggingFace and place the `.gguf` file in `models/`.

## Examples

- `lessons/01_basics/01_chat.py` — basic question/answer with Llama 3.
- `lessons/01_basics/04_chat_template.py` — same idea with an explicit chat template.

Run either after downloading the model:

```bash
python lessons/01_basics/01_chat.py
python lessons/01_basics/04_chat_template.py
```

Edit the `llama_path` / `MODEL_PATH` variable in each file if your model is elsewhere.

## Notes

- `n_gpu_layers=0` runs on CPU. On this Apple Silicon machine you can enable Metal by setting `n_gpu_layers` to a higher value (llama-cpp-python was built with Metal support by default on macOS).
- Adjust `n_threads` to match your core count for best CPU performance.