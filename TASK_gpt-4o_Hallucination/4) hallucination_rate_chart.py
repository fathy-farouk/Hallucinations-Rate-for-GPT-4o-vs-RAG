import pandas as pd
import matplotlib.pyplot as plt

# Load scored results
df_gpt = pd.read_csv("hallucination_scored_1000.csv")

# Count total and hallucinated
GPT_4o_count= df_gpt["hallucination_class"].value_counts()

# Get values
total = len(df_gpt)
grounded = GPT_4o_count.get("based on the context", 0)
hallucinated = GPT_4o_count.get("not based on the context", 0)
rate = (hallucinated / total) * 100

# Print stats
print("📊 Total Samples:", total)
print("✅ Grounded Answers:", grounded)
print("❌ Hallucinated Answers:", hallucinated)
print(f"🤯 Hallucination Rate: {rate:.2f}%")


# Prepare pie chart
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
plt.title("🤖 GPT_4o Answer Hallucination Rate", fontsize=14)
plt.tight_layout()
plt.show()
