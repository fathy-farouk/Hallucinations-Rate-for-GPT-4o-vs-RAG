import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import time

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever + LLM
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Load 1000 SQuAD samples
df = pd.read_csv("sampled_squad_5000.csv").head(1000)

# Generate answers
rag_answers = []
for i, row in df.iterrows():
    print(f"Processing {i+1}/1000")
    result = qa_chain.run(row["question"])
    rag_answers.append(result)
    time.sleep(1)  # Respect rate limits

# Save results
df["rag_answer"] = rag_answers
df.to_csv("rag_answers_1000.csv", index=False)
print("✅ Done! RAG answers saved to rag_answers_1000.csv")
