"""
Minimal example using llama-cpp-python with a GGUF model.

Prerequisites:
  1. pip install llama-cpp-python
  2. Download a GGUF model file, e.g.:
     https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
     -> Llama-3.2-3B-Instruct-Q4_K_M.gguf

Set the MODEL_PATH below to the path of your downloaded .gguf file.
"""

from llama_cpp import Llama

MODEL_PATH = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"  # <-- change me


def main() -> None:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,        # context window
        n_threads=4,       # CPU threads (adjust to your machine)
        n_gpu_layers=0,    # set >0 if you have Metal/CUDA support
        verbose=False,
    )

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "What is the capital of France?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    response = llm(
        prompt,
        max_tokens=128,
        temperature=0.7,
        top_p=0.95,
        stop=["<|eot_id|>"],
    )

    print(response["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()