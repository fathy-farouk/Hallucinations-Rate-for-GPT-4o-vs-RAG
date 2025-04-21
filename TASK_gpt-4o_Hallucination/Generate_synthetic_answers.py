import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
client = OpenAI()  # Auto-reads your OPENAI_API_KEY from .env

# Load the 5,000 sampled SQuAD rows
df = pd.read_csv("sampled_squad_5000.csv").head(1000)

# Define function to call GPT-4o using new SDK
def ask_gpt4o(context, question):
    prompt = f"""Use ONLY the following context to answer the question. 
If you don’t know the answer from the context, say "I don’t know".

Context:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Error:", e)
        return "ERROR"

# Generate synthetic answers
gpt_answers = []
for i, row in df.iterrows():
    print(f"Processing {i+1}/1000")
    answer = ask_gpt4o(row['context'], row['question'])
    gpt_answers.append(answer)
    time.sleep(1)  # Avoid OpenAI rate limits

# Save output
df["gpt4o_answer"] = gpt_answers
df.to_csv("squad_gpt4o_answers.csv", index=False)
print("✅ Finished! Saved to squad_gpt4o_answers.csv")
