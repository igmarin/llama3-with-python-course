import yaml

# Local configuration for Mac with llama-cpp-python
config_dict = {
    # Update the model (local GGUF file)
    "model": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "batch_size": 8,
    "device": "cpu",  # or "mps" for Apple Silicon
    "n_ctx": 2048,  # Context window size
    "n_gpu_layers": -1,  # Offload to GPU if available
    "verbose": False,
    "learning_rate": 3e-05,
    "output_dir": "./finetune_results"
}

# Save the updated configuration to a new YAML file
with open("local_config.yaml", "w") as yaml_file:
    yaml.dump(config_dict, yaml_file)

print("Configuration saved to local_config.yaml")
print("\nConfiguration:")
print(yaml.dump(config_dict, default_flow_style=False))

# To load it back:
# with open("local_config.yaml", "r") as yaml_file:
#     loaded_config = yaml.safe_load(yaml_file)
