from datasets import load_from_disk

# Save the dataset to disk
ds.save_to_disk("preprocessed_dataset")

# Load the dataset from disk
ds_preprocessed = load_from_disk("preprocessed_dataset")

# Print the first element of the loaded dataset
print(ds_preprocessed[0])
