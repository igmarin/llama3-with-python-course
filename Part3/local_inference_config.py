from llama_cpp import Llama
from datasets import Dataset
import pandas as pd

# Local configuration for Mac
config_dict = {
    # Define the model (local GGUF file)
    "model": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    # Define the batch size (for llama-cpp, this is context window)
    "n_ctx": 2048,
    # Define the device type (macOS uses CPU or MPS for Apple Silicon)
    "device": "cpu",  # or "mps" if you have Apple Silicon and metal support
    "n_gpu_layers": -1,  # -1 means offload all layers to GPU if available
    "verbose": False,
    # Training parameters (conceptual - llama-cpp doesn't do fine-tuning directly)
    "epochs": 15,
    "learning_rate": 3e-05,
    "output_dir": "./finetune_results"
}

print("Configuration for local Mac setup:")
for key, value in config_dict.items():
    print(f"  {key}: {value}")

# Initialize the model with config
llm = Llama(
    model_path=config_dict["model"],
    n_ctx=config_dict["n_ctx"],
    n_gpu_layers=config_dict["n_gpu_layers"],
    verbose=config_dict["verbose"]
)

print("\nModel loaded successfully!")

# Load dataset
df = pd.read_csv('data/email_categories_data.csv')
ds = Dataset.from_pandas(df)

print(f"Dataset loaded: {len(ds)} samples")

# Note: For actual fine-tuning on Mac with GGUF models, you would need:
# - peft + transformers (requires converting GGUF to HF format)
# - or use axolotl/llm-foundry with proper setup
# This config is for inference with your current setup
