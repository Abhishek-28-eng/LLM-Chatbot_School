from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Extract data and prepare embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define index parameters
index_name = "llm-chatbot"
dimension = 384  # Ensure this matches the embedding model's output dimension
metric = "cosine"

# Check if the index exists, create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",        # Specify cloud provider
            region="us-east-1"  # Specify region
        )
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")

# Fetch the host for the index
index_host = pc.describe_index(index_name).host

# Connect to the Pinecone index with the api_key, host, and index_name
pinecone_index = Index(
    name=index_name, 
    api_key=PINECONE_API_KEY,  # Pass the API key here
    host=index_host
)

# Store text chunks with embeddings
docsearch = LangchainPinecone.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name  # Pass index name here
)

print("Data has been indexed successfully!")
