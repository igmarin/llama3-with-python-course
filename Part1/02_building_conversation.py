"""Part 2: Building a conversation with Llama 3.

The model accepts a list of messages, each with a "role" and "content".
This example builds a customer-support style conversation and prints the
model's reply using create_chat_completion.
"""

from llama_cpp import Llama

llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

llm = Llama(
    model_path=llama_path,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

prompt = "Give me four short steps to troubleshoot my internet connection."

conv = [
    # Complete the user message
    {
        "role": "user",
        "content": prompt,
    }
]

# Pass the conversation to the model
result = llm.create_chat_completion(messages=conv, max_tokens=20)
print(result)