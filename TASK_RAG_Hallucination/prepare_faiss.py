import pandas as pd
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Load SQuAD data
df = pd.read_csv("sampled_squad_5000.csv").head(1000)  # Use same 1000 rows

import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


# Turn contexts into Document objects
documents = [Document(page_content=row["context"]) for _, row in df.iterrows()]

# Optional: Chunking (helps FAISS indexing)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed using OpenAI
embeddings = OpenAIEmbeddings()

# Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save for later
vectorstore.save_local("faiss_index")

print("✅ FAISS vectorstore created and saved.")
