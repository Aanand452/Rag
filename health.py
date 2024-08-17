# # This code snippet is performing the following tasks:
import os
import sys 
import pinecone
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the PDF document
loader = PyPDFDirectoryLoader("pdfs")
data = loader.load()

# Split the text into 500 chunks, do not overlap 40
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
text_chunks = text_splitter.split_documents(data)

# Import the OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(openai_api_key=OPENAI_API_KEY)

# Import the Pinecone API key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# Initialize Pinecone with the API key and environment
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create index name
index_name = "badluck"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings.embed_query ,
    index_name=index_name
)

# Index documents
docSearch = vector_store.from_texts([t.page_content for t in text_chunks],embedding=embeddings,index_name=index_name)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=client,
    chain_type="stuff",
    retriever=docSearch.as_retriever(),
    return_source_documents=True
)

# Query
query = "What is the summary of the document?"
response = qa.invoke(query)

response



while True:
    user_input = input(f"Input Prompt: ")
    if user_input == "exit":
        print("Exiting")
        sys.exit()
    if user_input == "":
        continue
    result = qa({'query':user_input})
    print(f"Answer:{result['result']}")
    