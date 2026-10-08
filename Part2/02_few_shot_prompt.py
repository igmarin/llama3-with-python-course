from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

prompt="""Review 1: I ordered from this place last night, and I'm impressed!
Sentiment 1: Positive,
Review 2: My order was delayed by over an hour without any updates. Disappointing!
Sentiment 2: Negative,
Review 3: The food quality is top-notch. Highly recommend!
Sentiment 3: Positive,
Review 4: Delicious food, and excellent customer service!
Sentiment 4:"""

# Send the prompt to the model with a stop word
output = llm(prompt, max_tokens=2, stop=["Review"])
print(output['choices'][0]['text'])
