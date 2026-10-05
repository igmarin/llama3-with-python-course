from llama_cpp import Llama

llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

llm = Llama(
    model_path=llama_path,
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

# Add a system message to the conversation list
conv = [
	{
        "role": "system",
        "content": "You are a helpful and professional customer support assistant for an internet service provider. If the question or instruction doesn't relate to internet service, quote the response: 'Sorry, I can't answer that.'"},
	{
        "role": "user",
	    "content": "Help me decide which stocks to invest in."
    }
]

result = llm.create_chat_completion(messages=conv, max_tokens=15)
# Extract the model response from the result object
assistant_content = result["choices"][0]["message"]["content"]
print(assistant_content)
