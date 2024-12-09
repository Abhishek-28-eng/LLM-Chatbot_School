import os
from dotenv import load_dotenv
import pinecone

# Load environment variables
load_dotenv()

# Get Pinecone API details from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = "llm-chatbot"  # Replace with your index name

# Initialize Pinecone client using the Pinecone class
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

def delete_all_vectors(index_name: str):
    """
    Deletes all vectors from the specified Pinecone index but keeps the index.

    Args:
        index_name (str): Name of the Pinecone index.
    """
    try:
        # Check if the index exists
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' does not exist.")
            return

        # Connect to the index
        index = pc.Index(index_name)

        # Delete all vectors
        index.delete(delete_all=True)
        print(f"All vectors deleted from index '{index_name}' while keeping the index intact.")

    except Exception as e:
        print(f"Error deleting vectors: {e}")

if __name__ == "__main__":
    delete_all_vectors(INDEX_NAME)
