from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from csv_handler import save_to_csv  # Import the CSV saving function
import os
#from ocr_handler import extract_text_with_ocr  # Import the OCR function
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
import numpy as np
import joblib  # For saving the PCA model

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get Pinecone API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define the index name
index_name = "llm-chatbot"

# Load the existing Pinecone index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Set up chain parameters
chain_type_kwargs = {"prompt": PROMPT}

# Load the LLM model
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Initialize the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

# Folder where your PDFs are located
DATA_FOLDER = "data"  # Change to the actual path if necessary

# Ensure the folder exists
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

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

# Route for rendering the chat interface
@app.route("/")
def index():
    """Render the chat interface."""
    return render_template('chat.html')

# Route to handle user input and generate a response using RetrievalQA
@app.route("/get", methods=["POST"])
def chat():
    """
    Handle user input and generate a response using RetrievalQA.
    Save the interaction in a CSV file.
    """
    try:
        # Get user input
        msg = request.form["msg"]
        print("Input:", msg)

        # Get the response from the model
        result = qa.invoke({"query": msg})
        response = result["result"]
        print("Response:", response)

        # Save the input and response to CSV
        save_to_csv(msg, response)

        # Return the response
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request."})

# Route to process a PDF from the 'data' folder using OCR
@app.route("/process_pdf/<filename>", methods=["GET"])
def process_pdf(filename):
    """
    Process a specific PDF from the 'data' folder using OCR.
    Args:
    - filename: Name of the PDF file to be processed.
    """
    try:
        # Check if the file exists in the data folder
        file_path = os.path.join(DATA_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"The file {filename} does not exist."})

        # Perform OCR on the PDF
        extracted_text = extract_text_with_ocr(file_path)
        if extracted_text:
            return jsonify({"filename": filename, "extracted_text": extracted_text})
        else:
            return jsonify({"error": "Failed to extract text from the PDF."})
    
    except Exception as e:
        print(f"Error processing PDF {filename}: {e}")
        return jsonify({"error": "An error occurred during OCR processing."})

# Route to upload documents, perform PCA, and then upload to Pinecone
@app.route("/upload_documents", methods=["POST"])
def upload_documents():
    """
    Upload documents, perform PCA on their embeddings, and store in Pinecone.
    """
    try:
        documents = request.json.get("documents", [])
        if not documents:
            return jsonify({"error": "No documents provided."})

        # Generate embeddings (you would use your existing function for this)
        embeddings = [download_hugging_face_embeddings(doc) for doc in documents]

        # Perform PCA on the embeddings
        reduced_embeddings = perform_pca(embeddings)

        # Upload the reduced embeddings to Pinecone (assuming you have a function to do this)
        for i, embedding in enumerate(reduced_embeddings):
            docsearch.index.upsert([{
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": {"text": documents[i]}
            }])

        return jsonify({"message": "Documents successfully uploaded and embeddings reduced."})

    except Exception as e:
        print(f"Error during document upload: {e}")
        return jsonify({"error": "An error occurred while uploading documents."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)