# Import evaluation libraries
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the local dataset
df = pd.read_csv('data/email_categories_data.csv')
dataset = Dataset.from_pandas(df)

# Create sample test answers (predictions) and reference answers
# For demonstration, we'll simulate predictions
# In a real scenario, these would be model outputs vs ground truth
test_answers = ["Priority", "Updates", "Promotions", "Priority", "Updates"]  # Simulated model predictions
reference_answers = dataset["category"][:5]  # Ground truth labels

print("Sample evaluation data:")
for i, (pred, ref) in enumerate(zip(test_answers, reference_answers)):
    print(f"\nExample {i+1}:")
    print(f"  Prediction: {pred}")
    print(f"  Reference: {ref}")

# Compute classification metrics
accuracy = accuracy_score(reference_answers, test_answers)
print("\n" + "=" * 50)
print("Classification Evaluation Results:")
print("=" * 50)
print(f"Accuracy: {accuracy:.2f}")

print("\nDetailed Classification Report:")
print(classification_report(reference_answers, test_answers))

print("\nConfusion Matrix:")
print(confusion_matrix(reference_answers, test_answers))

print("\n--- Notes ---")
print("For classification tasks (like email categories), sklearn metrics are appropriate:")
print("  - Accuracy: Overall correctness")
print("  - Precision: How many selected items are relevant")
print("  - Recall: How many relevant items are selected")
print("  - F1-score: Harmonic mean of precision and recall")
print("\nROUGE is for text generation tasks (summarization, translation)")
print("not for classification tasks with discrete labels.")
