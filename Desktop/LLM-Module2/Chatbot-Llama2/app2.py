from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from csv_handler import save_to_csv  # Import the CSV saving function
import os
from ocr_handler import extract_text_with_ocr  # Import the OCR function
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
import numpy as np
import joblib  # For saving the PCA model
import requests  # For Ollama integration

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get Pinecone API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define the index name
index_name = "llm-chatbot"

# Load the existing Pinecone index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define the prompt template for question answering
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to perform PCA and reduce embeddings' dimensions
def perform_pca(embeddings, components=128):
    """
    Reduces the dimensionality of embeddings using PCA.
    Args:
    - embeddings: List of embeddings (e.g., from HuggingFace model)
    - components: Number of PCA components to reduce to
    """
    pca = PCA(n_components=components)
    reduced_embeddings = pca.fit_transform(np.array(embeddings))
    
    # Save the PCA model for future use
    joblib.dump(pca, "pca_model.pkl")
    
    return reduced_embeddings

# Ollama API function (for llama3.2)
def query_ollama(prompt, model="llama3.2"):
    url = "http://127.0.0.1:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        # Split response into potential JSON objects
        responses = response.text.split("\n")
        for part in responses:
            try:
                json_response = json.loads(part)
                return json_response.get("completion", "No response found in the API output.")
            except json.JSONDecodeError:
                continue
        return "Error: No valid JSON found in the API response."
    
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return f"Error: Unable to connect to the API. Details: {e}"




# Route for rendering the chat interface
@app.route("/")
def index():
    """Render the chat interface."""
    return render_template('chat.html')

# Route to handle user input and generate a response using Ollama
@app.route("/get", methods=["POST"])
def chat():
    """
    Handle user input and generate a response using Ollama and Pinecone.
    Save the interaction in a CSV file.
    """
    try:
        # Get user input
        msg = request.form["msg"]
        print("Input:", msg)

        # Retrieve context from Pinecone
        context = retrieve_context(msg)

        # Create the prompt for Ollama
        prompt = f"Context:\n{context}\n\nQuestion:\n{msg}\n\nAnswer:"

        # Get the response from Ollama (llama3.2 model)
        response = query_ollama(prompt)
        print("Response:", response)

        # Save the input and response to CSV
        save_to_csv(msg, response)

        # Return the response as plain text
        return response

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing the request."

# Function to retrieve context from Pinecone
def retrieve_context(query):
    """
    Retrieve relevant context from Pinecone for the query.
    Args:
    - query: User input query
    Returns:
    - Context string
    """
    try:
        docs = docsearch.similarity_search(query, k=2)  
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "No relevant context found."

# Route to process a PDF from the 'data' folder using OCR
@app.route("/process_pdf/<filename>", methods=["GET"])
def process_pdf(filename):
    """
    Process a specific PDF from the 'data' folder using OCR.
    Args:
    - filename: Name of the PDF file to be processed.
    """
    try:
        # Define the data folder
        DATA_FOLDER = "data"

        # Check if the file exists in the data folder
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return f"The file {filename} does not exist."

        # Perform OCR on the PDF
        extracted_text = extract_text_with_ocr(file_path)
        if extracted_text:
            return f"Filename: {filename}\nExtracted Text:\n{extracted_text}"
        else:
            return "Failed to extract text from the PDF."
    
    except Exception as e:
        print(f"Error processing PDF {filename}: {e}")
        return "An error occurred during OCR processing."

# Main application runner
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
