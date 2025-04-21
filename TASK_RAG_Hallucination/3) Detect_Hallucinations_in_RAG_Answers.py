from transformers import pipeline
import pandas as pd

# Load your first 1000 answers
df = pd.read_csv("rag_answers_1000.csv")

# Load classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define hallucination detection logic
def is_hallucinated(context, answer):
    try:
        result = classifier(
            sequences=answer,
            candidate_labels=["based on the context", "not based on the context"],
            hypothesis_template="This answer is {}."
        )
        return result['labels'][0]
    except Exception as e:
        print("❌ Error:", e)
        return "error"

# Run on all RAG answers
labels = []
for i, row in df.iterrows():
    print(f"Checking {i+1}/1000")
    label = is_hallucinated(row["context"], row["rag_answer"])
    labels.append(label)

# Save with hallucination labels
df["hallucination_label"] = labels
df.to_csv("rag_hallucination_scored_1000.csv", index=False)
print("("✅ Done. Saved results to rag_hallucination_scored_1000.csv")
 
