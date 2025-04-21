from datasets import load_dataset
import random
import pandas as pd

# Load the SQuAD v1.1 training dataset
dataset = load_dataset("squad", split="train")

# Convert to list of dictionaries
data = [{"context": item["context"], "question": item["question"], "answer": item["answers"]["text"][0]} for item in dataset]

# Sample 5,000 randomly
sampled_data = random.sample(data, 5000)

# Save as DataFrame and export to CSV for inspection
df = pd.DataFrame(sampled_data)
df.to_csv("sampled_squad_5000.csv", index=False)

print("✅ Sampled 5,000 examples and saved to sampled_squad_5000.csv")
