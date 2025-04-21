import pandas as pd
import matplotlib.pyplot as plt

# Load RAG results only
df_rag = pd.read_csv("rag_hallucination_scored_1000.csv")

# Count hallucination labels
rag_counts = df_rag["hallucination_label"].value_counts()

# Get values
total = len(df_rag)
grounded = rag_counts.get("based on the context", 0)
hallucinated = rag_counts.get("not based on the context", 0)
rate = (hallucinated / total) * 100

# Print stats with emojis
print("📊 Total Samples:", total)
print("✅ Grounded Answers:", grounded)
print("❌ Hallucinated Answers:", hallucinated)
print(f"🤯 Hallucination Rate: {rate:.2f}%")

# Setup for pie chart
labels = ["based on the context", "not based on the context"]
colors = ["#66b3ff", "#ff6666"]
explode = [0.05, 0.05]

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    [grounded, hallucinated],
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode
)
plt.title("🤖 RAG Answer Hallucination Rate", fontsize=14)
plt.tight_layout()
plt.show()
