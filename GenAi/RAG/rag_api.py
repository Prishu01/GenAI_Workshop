import os 
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA 

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

url= "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)

users= response.json()

print("Users Loaded:", len(users))

documents = []
for user in users:

    text = f"""
    ID : {user['id']}
    Name : {user['name']}
    Username : {user['username']}
    Email : {user['email']}
    Phone : {user['phone']}
    Website : {user['website']}

    Company :
    {user['company']['name']}

    Catch Phrase :
    {user['company']['catchPhrase']}

    City :
    {user['address']['city']}

    Street :
    {user['address']['street']}
    """

    documents.append(
        Document(page_content=text)
    )

print("Documents Created:", len(documents))


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

docs = splitter.split_documents(documents)

print("chunks :", len(docs))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(
    docs,
    embeddings
)

print("FAISS Created")

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

llm=ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0
)

qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\nRAG Ready")

while True:
    question = input("\nAsk a question :")

    if question.lower() == "exit":
        break
    result = qa.invoke({"query": question})

    print("\nAnswer :")
    print(result['result'])

    print("\nRetrieved contexts :")

    for doc in result['source_documents']:
        print("......................................")
        print(doc.page_content)
        
