from llama_cpp import Llama
from datasets import Dataset
import pandas as pd

# Path to the downloaded GGUF model file.
llama_path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

# Load the local CSV dataset
df = pd.read_csv('data/email_categories_data.csv')
ds = Dataset.from_pandas(df)

# Filter for the first 500 samples of the dataset
filtered_ds = ds.select(range(min(500, len(ds))))

print(f"Loaded {len(filtered_ds)} samples from local dataset")
print(f"First sample: {filtered_ds[0]}")
