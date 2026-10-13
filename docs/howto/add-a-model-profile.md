# Add a model profile

Make a new model selectable by name. This example adds Llama 3.2 1B, a tiny model that loads fast and
uses little memory. The same steps work for any GGUF model on Hugging Face, the website where most
open models are published.

## Steps

1. Find a repository with GGUF files and pick one file. Search Hugging Face for `GGUF`. Many
   people publish compressed (quantized) copies of models; the `bartowski` repositories list every
   file and its size. Choose a single `.gguf` file; models split into parts
   (`...-00001-of-00002.gguf`) will not work with the download script.

2. Add a block under `profiles:` in [`configs/models.yaml`](../../configs/models.yaml):

   ```yaml
     tiny-1b:
       repo: bartowski/Llama-3.2-1B-Instruct-GGUF
       file: Llama-3.2-1B-Instruct-Q4_K_M.gguf
       n_ctx: 4096
       n_gpu_layers: -1
       n_threads: null
   ```

   | Field | Meaning |
   |-------|---------|
   | `repo`, `file` | Where to download from, and which file |
   | `n_ctx` | Context window in tokens. Larger needs more memory |
   | `n_gpu_layers` | How many of the model's layers (its stacked processing stages) run on the GPU. `-1` means all of them, using Metal on a Mac |
   | `n_threads` | How many CPU threads to use. `null` lets llama.cpp choose |
   | `mlx_repo` | Optional. The same model in MLX format, needed only for [lesson 7](../../lessons/07_finetuning/) fine-tuning |

3. Download it:

   ```bash
   python scripts/download_model.py --profile tiny-1b
   ```

4. Check that it loads and run a lesson:

   ```bash
   python -c "from llama_course.llm import get_profile; print(get_profile('tiny-1b'))"
   LLAMA_PROFILE=tiny-1b python lessons/01_basics/03_system_prompt.py
   ```

   You see the profile printed, then the shop's refusal: `I can only help with questions about
   bikes.` (This was run with this exact profile.)

5. Measure it:

   ```bash
   python scripts/benchmark.py --profile tiny-1b
   ```

   On a 16 GB MacBook Pro this showed 104 tokens per second and 1.1 GB of peak memory, against 45
   tokens per second and 2.6 GB for the 3B model.

## Notes

- The `tiny-1b` block is not in the shipped [`configs/models.yaml`](../../configs/models.yaml). Add it yourself to follow along.
- Profile names are free text. Use names that say what the machine or model is.
- A model from another family (Mistral, Qwen) should load the same way (not tested here), but its prompt format differs,
  and `llama3_chat_prompt` in [lesson 1.4](../../lessons/01_basics/04_chat_template.py) is specific to Llama 3.
