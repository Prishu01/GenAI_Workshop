import os
import json
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
# Load JSON File
# -----------------------------
with open("C:\\Users\\prish\\OneDrive\\Desktop\\RAG_WORKSHOP\\data\\student.json", "r", encoding="utf-8") as file:
    students = json.load(file)

print(f"Students Loaded: {len(students)}")

# -----------------------------
# Convert JSON -> Documents
# -----------------------------
documents = []

for student in students:

    text = f"""
Student ID : {student['id']}

Name : {student['name']}

Course : {student['course']}

Marks : {student['marks']}
"""

    documents.append(
        Document(
            page_content=text,
            metadata={
                "id": student["id"],
                "name": student["name"],
                "course": student["course"],
                "marks": student["marks"]
            }
        )
    )

print(f"Documents Created: {len(documents)}")

# -----------------------------
# Split Documents
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)

docs = splitter.split_documents(documents)

print(f"Chunks Created: {len(docs)}")

# -----------------------------
# Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create FAISS Index
# -----------------------------
vector_db = FAISS.from_documents(
    docs,
    embeddings
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
    temperature=0
)

# -----------------------------
# Retrieval QA
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
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

    print("\nAnswer:")
    print(result["result"])

    print("\nRetrieved Documents:\n")

    for doc in result["source_documents"]:
        print("=" * 60)
        print(doc.page_content)