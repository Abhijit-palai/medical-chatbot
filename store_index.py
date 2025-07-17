from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()


#initialize the pinecone

from pinecone import Pinecone, ServerlessSpec, PodSpec # Make sure PodSpec is imported if you might use it

# Initialize Pinecone
# Replace with your actual API key and environment
# Ensure your environment matches the region you plan to use if it's region-specific
PINECONE_API_KEY = PINECONE_API_KEY
PINECONE_API_ENV = PINECONE_API_ENV # e.g., "us-east-1-aws" or similar, check your dashboard

pinecone = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name = "medical-chatbot"

# Check if the index exists
# pinecone.list_indexes() returns a list of dictionaries, so we need to check names
existing_indexes = pinecone.list_indexes()
existing_index_names = [idx['name'] for idx in existing_indexes]

if index_name not in existing_index_names:
    print(f"Creating index '{index_name}'...")
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your embedding model's dimension
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1') # Or your free-tier supported region
    )
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists. Connecting to it.")

# Now you can connect to your index
index = pinecone.Index(index_name)


from langchain_pinecone import Pinecone as LangChainPinecone # Alias LangChain's Pinecone class
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document # Import if your text_chunks are Document objects

# --- Assume these variables are already correctly defined and initialized from previous steps ---
# PINECONE_API_KEY
# PINECONE_API_ENV
# index_name
# text_chunks (e.g., list of Document objects with .page_content)
# embeddings (your HuggingFaceEmbeddings or OpenAIEmbeddings instance)
# ---------------------------------------------------------------------------------------------

import os

os.environ["PINECONE_API_KEY"] = "pcsk_4ox7EQ_48i9bJpVkFXjr1qGS1xRptLkMiiMQtUTkxVEXiQDmohCXvBAfRNu7q9CtXvRTG5"
os.environ["PINECONE_API_ENV"] = "your-env"  # e.g., "gcp-starter"


# Ensure the embedding model is initialized
# This is usually done once at the top level
# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

index_name = "medical-chatbot"

# --- Create/Load Vectorstore using LangChain's Pinecone integration ---
print(f"Attempting to create/load Pinecone vectorstore for index '{index_name}' using LangChain...")

try:
    # Use LangChainPinecone (the aliased class from langchain_pinecone)
    # This class handles embedding the text_chunks and upserting them to Pinecone
    docsearch = LangChainPinecone.from_texts(
        [t.page_content for t in text_chunks], # Ensure this extracts pure strings if text_chunks are Document objects
        embeddings,
        index_name=index_name,
        # If your LangChainPinecone version requires it, you might also need to pass
        # api_key="pcsk_4ox7EQ_48i9bJpVkFXjr1qGS1xRptLkMiiMQtUTkxVEXiQDmohCXvBAfRNu7q9CtXvRTG5",
        # environment=PINECONE_API_ENV
        # However, it often picks these up from environment variables or the global pinecone.init()
        # for the latest versions of langchain-pinecone.
    )
    print("Pinecone vectorstore created/loaded successfully using LangChain.")

except Exception as e:
    print(f"ERROR: Failed to create/load Pinecone vectorstore with LangChain: {e}")
    # Provide more specific guidance for common issues
    print("Possible reasons:")
    print("1. 'text_chunks' format: Ensure it's a list of strings or list of Document objects with '.page_content'.")
    print("2. Environment/Kernel Issue: If 'AttributeError: from_texts...' persists, restart your Python kernel.")
    print("3. API Key/Environment: Double-check that PINECONE_API_KEY and PINECONE_API_ENV are correct and accessible.")
    print("4. Index Dimension Mismatch: The dimension of your embeddings must match the dimension of your Pinecone index.")
    raise # Re-raise the exception for full traceback


