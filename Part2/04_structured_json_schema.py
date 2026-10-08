from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

messages = [
    {
        "role": "system",
        "content": "You are a helpful tutor for an online education platform. Answer student questions clearly and concisely."
    },
    {
        "role": "user",
        "content": "What is the difference between a list and a tuple in Python?"
    }
]

output = llm.create_chat_completion(
    messages=messages,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"Question": {"type": "string"}, "Answer": {"type": "string"}}
        }
    }
)

print(output['choices'][0]['message']['content'])
