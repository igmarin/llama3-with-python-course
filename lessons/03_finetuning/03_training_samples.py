from llama_cpp import Llama
from datasets import Dataset
import pandas as pd

# Path to the downloaded GGUF model file.
llama_path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

# Load the local CSV dataset
df = pd.read_csv('data/email_categories_data.csv')
dataset = Dataset.from_pandas(df)

def create_intent_example(row):
    # Fill out the columns in the prompt
    row['intent_example'] = f"Query: {row['email']}\nIntent: {row['category']}"
    return row

# Call the ds method to apply our preprocessing function to all rows
processed_dataset = dataset.map(create_intent_example)

# Print the intent_example in the first row of the processed data
print(processed_dataset[0]['intent_example'])
print("\n--- Second example ---")
print(processed_dataset[1]['intent_example'])
