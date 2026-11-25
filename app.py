
from flask import Flask, render_template, jsonify, request  # Importing Flask framework and related modules for web handling
from dotenv import load_dotenv  # For loading environment variables from .env file
from langchain_pinecone import Pinecone  # LangChain's Pinecone integration for vector database operations

from langchain_community.embeddings import HuggingFaceEmbeddings  # Import embedding model from HuggingFace
from langchain.prompts import PromptTemplate  # For creating structured prompt templates
from langchain_community.llms import CTransformers  # Load local LLM (like LLaMA) using CTransformers
from langchain.chains import RetrievalQA  # Retrieval-based Question Answering chain from LangChain
from pinecone import Pinecone as PineconeClient, ServerlessSpec  # Pinecone client for database operations
from src.helper import download_hugging_face_embeddings  # Custom helper function to download embeddings
from src.prompt import *  # Import custom prompt template from the prompt.py file
import os  # OS module for environment and file handling

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Load credentials from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')  # Retrieves Pinecone API key
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')  # Retrieves Pinecone environment (region info)

# Define Pinecone index name
index_name = "medical-chatbot"

# Initialize embeddings using the helper function
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client using the API key
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Get list of existing indexes in Pinecone account
existing_indexes = pc.list_indexes().names()

# Check if the desired index already exists or not
if index_name not in existing_indexes:
    print(f"Creating index '{index_name}'...")  # Log message if index doesn't exist
    pc.create_index(
        name=index_name,  # Name of the new index
        dimension=1536,  # Dimension of embedding vectors (must match your model)
        metric='cosine',  # Similarity metric used to compare embeddings
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Serverless configuration for AWS region
    )
    print("Index created.")  # Confirmation message
else:
    print(f"Index '{index_name}' already exists.")  # Log message if index already exists

# Connect to the existing or newly created Pinecone index
index = pc.Index(index_name)

# Wrap Pinecone index with LangChain’s vectorstore interface
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,  # Connect to existing index name
    embedding=embeddings,  # Embedding model used for vector search
    namespace="text"  # Optional namespace to organize vectors (can be removed)
)

# Create a retriever to fetch relevant information from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Retrieves top 2 relevant chunks per query

# Define the prompt template to control how model interacts with user query
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Uses imported template text

# Extra configurations for the retrieval chain
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the LLM (LLaMA or similar) using CTransformers with configuration
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # Local path to the quantized LLaMA model file
    model_type="llama",  # Defines type of model
    config={
        'max_new_tokens': 512,  # Maximum number of tokens generated in a single response
        'temperature': 0.8  # Controls randomness/creativity in the response
    }
)

# Create a Retrieval-based Question Answering (QA) chain
qa = RetrievalQA.from_chain_type(
    llm=llm,  # The loaded LLM used for answering
    chain_type="stuff",  # Combines retrieved documents (“stuffing”) before sending to the LLM
    retriever=retriever,  # Retriever responsible for fetching context
    return_source_documents=True,  # Enables returning the source documents used for context
    chain_type_kwargs=chain_type_kwargs  # Additional settings including the custom prompt
)

# Flask route for home page
@app.route("/")
def index():
    return render_template("chat.html")  # Renders the frontend HTML chat interface

###add
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Extracts JSON data sent from frontend
    query = data.get("query", "")  # Retrieves the user question from JSON payload

    if not query:  # Checks if user didn’t provide a question
        return jsonify({"response": "Please provide a query."}), 400  # Returns error message with HTTP 400 (Bad Request)

    try:
        result = qa({"query": query})  # Passes the query through the QA chain to get answer
        response = result["result"]  # Extracts the generated answer text
        return jsonify({"response": response})  # Sends back response in JSON format
    except Exception as e:  # Catches any errors during process
        print("Error during QA:", e)  # Logs the error in console
        return jsonify({"response": "Sorry, something went wrong."}), 500  # Sends error message with HTTP 500

# Runs the Flask app
if __name__ == '__main__':
    app.run(debug=True)  # Starts Flask server in debug mode for live reload and error tracing
