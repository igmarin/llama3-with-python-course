from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

# Add formatting to the prompt
prompt="""
Instruction: Explain the concept of gravity in simple terms.
Question: What is gravity?
Answer:
"""

# Send the prompt to the model
output = llm(prompt, max_tokens=15, stop=["Question:"])
print(output['choices'][0]['text'])
