from src.helper import load_pdf,text_split,download_hugging_face_embeddings  # Importing three helper functions (load_pdf, text_split, download_hugging_face_embeddings) from the src.helper module.
from langchain.vectorstores import Pinecone                                # Importing Pinecone vectorstore class from langchain.vectorstores (used to interact with Pinecone through LangChain).
import pinecone                                                           # Importing the pinecone client library to manage Pinecone operations directly.
from dotenv import load_dotenv                                            # Importing load_dotenv to load environment variables from a .env file.
import os                                                                 # Importing the os module to interact with environment variables and filesystem.

load_dotenv()                                                             # Load environment variables from a .env file into os.environ.

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')                        # Read the PINECONE_API_KEY value from environment variables and store it in PINECONE_API_KEY.
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')                        # Read the PINECONE_API_ENV value from environment variables and store it in PINECONE_API_ENV.

# print(PINECONE_API_KEY)                                                  # A commented-out debug print that, if uncommented, would display the Pinecone API key (avoid printing secrets in production).
# print(PINECONE_API_ENV)                                                  # A commented-out debug print that, if uncommented, would display the Pinecone environment.

extracted_data=load_pdf("data/")                                           # Call the load_pdf function (from src.helper) with "data/" directory to load PDF documents and store the result in extracted_data.
text_chunks=text_split(extracted_data)                                     # Call the text_split function (from src.helper) to split loaded documents into chunks; store chunks in text_chunks.
embeddings=download_hugging_face_embeddings()                              # Call the download_hugging_face_embeddings function to get an embeddings object and store it in embeddings.

#initialize the pinecone

from pinecone import Pinecone, ServerlessSpec, PodSpec # Make sure PodSpec is imported if you might use it  # Import specific classes (Pinecone client, ServerlessSpec, PodSpec) from the pinecone package; helpful for creating indexes with specific specs.

# Initialize Pinecone
# Replace with your actual API key and environment
# Ensure your environment matches the region you plan to use if it's region-specific
PINECONE_API_KEY = PINECONE_API_KEY                                              # Reassigning the PINECONE_API_KEY variable to itself (no-op here but keeps the name in local scope).
PINECONE_API_ENV = PINECONE_API_ENV                                              # Reassigning the PINECONE_API_ENV variable to itself (no-op here but keeps the name in local scope). # e.g., "us-east-1-aws" or similar, check your dashboard

pinecone = Pinecone(                                                              # Instantiate a Pinecone client object using the imported Pinecone class and assign it to the name 'pinecone' (this shadows the earlier pinecone import variable).
    api_key=PINECONE_API_KEY,                                                     # Provide the API key parameter to the Pinecone constructor using the PINECONE_API_KEY value.
    environment=PINECONE_API_ENV                                                  # Provide the environment/region parameter to the Pinecone constructor using the PINECONE_API_ENV value.
)

index_name = "medical-chatbot"                                                   # Define the index name to be used in Pinecone as the string "medical-chatbot".

# Check if the index exists
# pinecone.list_indexes() returns a list of dictionaries, so we need to check names
existing_indexes = pinecone.list_indexes()                                       # Call the list_indexes() method on the Pinecone client to retrieve a list of existing indexes and store in existing_indexes.
existing_index_names = [idx['name'] for idx in existing_indexes]                # Extract the 'name' field from each index dictionary into a list existing_index_names.

if index_name not in existing_index_names:                                      # Check whether the target index_name is absent from the existing index names.
    print(f"Creating index '{index_name}'...")                                   # Print a message indicating creation of the new index.
    pinecone.create_index(                                                       # Call create_index on the Pinecone client to create a new index with the specified parameters.
        name=index_name,                                                         # Set the index name to the value of index_name.
        dimension=1536,  # Ensure this matches your embedding model's dimension # Set the vector dimension for the index (must match embedding dimension).
        metric='cosine',                                                         # Set the similarity metric for the index to 'cosine'.
        spec=ServerlessSpec(cloud='aws', region='us-east-1') # Or your free-tier supported region # Provide a ServerlessSpec for the index specifying cloud provider and region.
    )
    print(f"Index '{index_name}' created successfully.")                         # Print a confirmation message after successful index creation.
else:
    print(f"Index '{index_name}' already exists. Connecting to it.")            # If the index exists, print that you'll connect to the existing index.

# Now you can connect to your index
index = pinecone.Index(index_name)                                              # Create an Index object that references the named index on Pinecone for upsert/query operations.

from langchain_pinecone import Pinecone as LangChainPinecone # Alias LangChain's Pinecone class  # Import LangChain's Pinecone wrapper and alias it to LangChainPinecone to avoid name collision with the pinecone client.
from langchain.embeddings import HuggingFaceEmbeddings                         # Import HuggingFaceEmbeddings from LangChain (if you plan to use LangChain's embedding wrapper).
from langchain_core.documents import Document # Import if your text_chunks are Document objects  # Import Document class from langchain_core.documents, helpful if text_chunks are Document instances rather than plain strings.

# --- Assume these variables are already correctly defined and initialized from previous steps ---
# PINECONE_API_KEY
# PINECONE_API_ENV
# index_name
# text_chunks (e.g., list of Document objects with .page_content)
# embeddings (your HuggingFaceEmbeddings or OpenAIEmbeddings instance)
# ---------------------------------------------------------------------------------------------

import os                                                                         # Re-importing os (already imported above) to emphasize environment variable usage.

os.environ["PINECONE_API_KEY"] = "pcsk_4ox7EQ_48i9bJpVkFXjr1qGS1xRptLkMiiMQtUTkxVEXiQDmohCXvBAfRNu7q9CtXvRTG5"  # Setting the PINECONE_API_KEY environment variable directly in the process (note: avoid hardcoding secrets in production).
os.environ["PINECONE_API_ENV"] = "your-env"  # e.g., "gcp-starter"                                            # Setting the PINECONE_API_ENV environment variable directly (placeholder value).


# Ensure the embedding model is initialized
# This is usually done once at the top level
# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

index_name = "medical-chatbot"                                                   # Re-declare the index_name variable to "medical-chatbot" (keeps the name in local scope for subsequent operations).

# --- Create/Load Vectorstore using LangChain's Pinecone integration ---
print(f"Attempting to create/load Pinecone vectorstore for index '{index_name}' using LangChain...")  # Print an informational message indicating the attempt to create/load the vectorstore.

try:
    # Use LangChainPinecone (the aliased class from langchain_pinecone)
    # This class handles embedding the text_chunks and upserting them to Pinecone
    docsearch = LangChainPinecone.from_texts(                                    # Call the LangChainPinecone.from_texts classmethod to create/load a vectorstore from a list of texts.
        [t.page_content for t in text_chunks], # Ensure this extracts pure strings if text_chunks are Document objects  # Provide a list comprehension that extracts .page_content from each Document in text_chunks (or fails if text_chunks are plain strings).
        embeddings,                                                               # Pass the embeddings object to be used to convert texts into vectors.
        index_name=index_name,                                                    # Specify the target Pinecone index name to store the vectors.
        # If your LangChainPinecone version requires it, you might also need to pass
        # api_key="pcsk_4ox7EQ_48i9bJpVkFXjr1qGS1xRptLkMiiMQtUTkxVEXiQDmohCXvBAfRNu7q9CtXvRTG5",
        # environment=PINECONE_API_ENV
        # However, it often picks these up from environment variables or the global pinecone.init()
        # for the latest versions of langchain-pinecone.
    )
    print("Pinecone vectorstore created/loaded successfully using LangChain.")    # Print success message when the LangChain Pinecone vectorstore is created/loaded.

except Exception as e:                                                          # Catch any exception raised during the vectorstore creation/loading process.
    print(f"ERROR: Failed to create/load Pinecone vectorstore with LangChain: {e}")  # Print an error message including the exception text.
    # Provide more specific guidance for common issues
    print("Possible reasons:")                                                   # Print header for troubleshooting tips.
    print("1. 'text_chunks' format: Ensure it's a list of strings or list of Document objects with '.page_content'.")  # Tip: check text_chunks format.
    print("2. Environment/Kernel Issue: If 'AttributeError: from_texts...' persists, restart your Python kernel.")     # Tip: kernel restart if attribute errors occur.
    print("3. API Key/Environment: Double-check that PINECONE_API_KEY and PINECONE_API_ENV are correct and accessible.") # Tip: verify API keys and env.
    print("4. Index Dimension Mismatch: The dimension of your embeddings must match the dimension of your Pinecone index.") # Tip: ensure embedding dim matches index dim.
    raise # Re-raise the exception for full traceback                                                                                  # Re-raise the caught exception to provide full traceback to the caller.
