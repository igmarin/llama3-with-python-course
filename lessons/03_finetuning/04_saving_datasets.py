from datasets import Dataset, load_from_disk
import pandas as pd
import os

# Load the local CSV dataset
df = pd.read_csv('data/email_categories_data.csv')
ds = Dataset.from_pandas(df)

# Preprocess the dataset
def create_intent_example(row):
    row['intent_example'] = f"Query: {row['email']}\nIntent: {row['category']}"
    return row

processed_dataset = ds.map(create_intent_example)

print("Original dataset size:", len(ds))
print("Processed dataset size:", len(processed_dataset))

# Save the dataset to disk
save_path = "preprocessed_dataset"
processed_dataset.save_to_disk(save_path)
print(f"\nDataset saved to {save_path}")

# Load the dataset from disk
ds_preprocessed = load_from_disk(save_path)
print("Dataset loaded from disk")

# Print the first element of the loaded dataset
print("\n--- First element of loaded dataset ---")
print(ds_preprocessed[0])

# Clean up (optional)
# import shutil
# shutil.rmtree(save_path)
