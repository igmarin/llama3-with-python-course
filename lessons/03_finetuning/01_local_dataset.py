from llama_cpp import Llama
from datasets import Dataset, load_dataset
import pandas as pd

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

# ============================================
# OPTION 1: Use a local CSV file (if you have one)
# ============================================
try:
    # Try loading from local CSV
    local_df = pd.read_csv('data/email_categories_data.csv')
    local_ds = Dataset.from_pandas(local_df)
    filtered_ds = local_ds.select(range(min(500, len(local_ds))))
    print(f"Loaded from local CSV: {len(filtered_ds)} samples")
except FileNotFoundError:
    print("Local CSV not found, using in-memory example...")

    # ============================================
    # OPTION 2: Create an in-memory dataset (works offline)
    # ============================================
    sample_data = {
        "text": [
            "I love this product! It's amazing.",
            "Terrible experience, would not recommend.",
            "Great quality and fast shipping.",
            "Worst purchase I've ever made.",
            "Absolutely fantastic, highly recommend!",
            "Disappointed with the quality.",
            "Perfect for my needs, thank you!",
            "Not worth the money.",
            "Excellent customer service.",
            "Poor packaging, item arrived damaged."
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    }

    local_ds = Dataset.from_dict(sample_data)
    filtered_ds = local_ds.select(range(min(500, len(local_ds))))
    print(f"Created in-memory dataset: {len(filtered_ds)} samples")

# ============================================
# OPTION 3: Load from Hugging Face (requires internet)
# ============================================
# Uncomment to use a real dataset:
# dataset_name = "imdb"
# ds = load_dataset(dataset_name, split="train")
# filtered_ds = ds.select(range(500))

# ============================================
# Test with the loaded dataset
# ============================================
print(f"\nDataset size: {len(filtered_ds)}")
print(f"First sample: {filtered_ds[0]}")
print(f"Columns: {filtered_ds.column_names}")

# Example: Classify a sample with Llama
if "text" in filtered_ds.column_names:
    sample_text = filtered_ds[0]["text"]
    prompt = f"Classify the sentiment as positive or negative: {sample_text}"
    output = llm(prompt, max_tokens=5)
    print(f"\nSample: {sample_text}")
    print(f"Classification: {output['choices'][0]['text'].strip()}")
