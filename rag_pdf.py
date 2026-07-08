import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

groq_key=os.getenv("GROQ_API_KEY")

loader = PyPDFLoader(
    "C:/Users/prish/OneDrive/Desktop/RAG_WORKSHOP/data/company_policy.pdf"
)
documents= loader.load()

print(f"\nTotal page loaded: {len(documents)}")

splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs=splitter.split_documents(documents)
print("chunks : ",len(docs))

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_db=FAISS.from_documents(docs, embedding)

retriever=vector_db.as_retriever(search_kwargs={"k":3})

llm=ChatGroq(
    groq_api_key=groq_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

qa=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)
print("\nRAG READY")

while True:
    question=input("\nAsk:")
    if question.lower() == "exit":
        break
    result=qa.invoke({"query":question})
    print("Answer:", result["result"])
    print("\nRetrieved Sources:")

    for i, doc in enumerate(result["source_documents"]):
        print("--------------------------------")
        print(doc.page_content)