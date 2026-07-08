import os
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Load API Data
# -----------------------------
url = "https://dummyjson.com/posts"

response = requests.get(url)
response.raise_for_status()

posts = response.json()["posts"]

print(f"Posts Loaded: {len(posts)}")

# -----------------------------
# Convert to Documents
# -----------------------------
documents = []

for post in posts:

    text = f"""
Post ID: {post['id']}

Title:
{post['title']}

Body:
{post['body']}

Tags:
{", ".join(post['tags'])}

Likes:
{post['reactions']['likes']}

Dislikes:
{post['reactions']['dislikes']}

Views:
{post['views']}

User ID:
{post['userId']}
"""

    documents.append(
        Document(
            page_content=text,
            metadata={
                "id": post["id"],
                "userId": post["userId"],
                "title": post["title"],
            },
        )
    )

print(f"Documents Created: {len(documents)}")

# -----------------------------
# Split Documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = splitter.split_documents(documents)

print(f"Chunks Created: {len(docs)}")

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# FAISS
# -----------------------------
vector_db = FAISS.from_documents(
    docs,
    embeddings,
)

print("FAISS Index Created")

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------
# Groq LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# -----------------------------
# Retrieval QA
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

print("\nRAG Ready!")

# -----------------------------
# Chat Loop
# -----------------------------
while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        break

    result = qa.invoke({"query": question})

    print("\nAnswer:\n")
    print(result["result"])

    print("\nRetrieved Documents:\n")

    for doc in result["source_documents"]:
        print("=" * 60)
        print(doc.page_content)