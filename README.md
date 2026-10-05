# Llama3 with Python

Minimal example using [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) to run a Llama 3 GGUF model locally on CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the dependencies for this project:

```
llama-cpp-python
huggingface-hub
```

- `llama-cpp-python` — the core library for loading and running GGUF models.
- `huggingface-hub` — used by `download_model.py` to fetch model files.

## Download a model

A small GGUF model is recommended for CPU-only use. Use the helper script:

```bash
python download_model.py --model bartowski/Llama-3.2-3B-Instruct-GGUF \
    --file Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Or download manually from HuggingFace and place the `.gguf` file in `models/`.

## Examples

- `Part1/chatting_llama.py` — basic question/answer with Llama 3.
- `example.py` — same idea with an explicit chat template (system/user/assistant).

Run either after downloading the model:

```bash
python Part1/chatting_llama.py
python example.py
```

Edit the `llama_path` / `MODEL_PATH` variable in each file if your model is elsewhere.

## Notes

- `n_gpu_layers=0` runs on CPU. On this Apple Silicon machine you can enable Metal by setting `n_gpu_layers` to a higher value (llama-cpp-python was built with Metal support by default on macOS).
- Adjust `n_threads` to match your core count for best CPU performance.