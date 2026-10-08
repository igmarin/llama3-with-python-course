# Import required libraries
import pandas as pd
from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)

# Load the email dataset
emails_df = pd.read_csv('data/email_categories_data.csv')
# Display the first few rows of our dataset
print("Preview of our email dataset:")
emails_df.head(2)

# Create the prompt for email classification
prompt = "Classify the following email into one of these categories: Priority, Updates, or Promotions. Email: {}"

# Test the model on the first two emails
email1 = emails_df.iloc[0, 0]
email2 = emails_df.iloc[1, 0]

# Classify first email
output1 = llm(prompt.format(email1), max_tokens=5)
result1 = output1['choices'][0]['text'].strip()

# Classify second email
output2 = llm(prompt.format(email2), max_tokens=5)
result2 = output2['choices'][0]['text'].strip()

print(result1)
print(result2)
