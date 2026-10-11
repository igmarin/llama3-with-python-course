"""Part 1: Chatting with Llama 3 locally using llama-cpp-python.

This is the corrected version of the starter snippet:
  - Import the `Llama` class (capital L), not the `llama` submodule.
  - Initialize with model_path=llama_path.
"""

from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the Llama model
llm = Llama(
    model_path=llama_path,
    n_ctx=4096,        # context window
    n_threads=4,       # CPU threads (adjust to your machine)
    n_gpu_layers=0,    # set >0 if you have Metal/CUDA support
    verbose=False,
)

question = "What is the most used database for data storage?"
response = llm(
    question,
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    stop=["<|eot_id|>"],
)

print(response["choices"][0]["text"].strip())