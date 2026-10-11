from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You convert inventory lists from text to JSON, extracting item counts and names from the text as keys and values in the form: item: count; for example, 'banana': 32.",},
            {"role": "user", "content": "Fifteen apples, thirty-three oranges, and five thousand fifty-two potatoes."},
        ],
        # Specify output format to JSON
        response_format={
            "type": "json_object"
        }
)

print(output['choices'][0]['message']['content'])
