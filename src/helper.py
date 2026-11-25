#from langchain.document_loaders import PyPDFLoader,DirectoryLoader        # (Old import) This line imports PyPDFLoader and DirectoryLoader from LangChain's document_loaders module, but it's commented out.

from langchain.text_splitter import RecursiveCharacterTextSplitter         # Imports a text splitter that divides long text into smaller overlapping chunks for better processing.

#from langchain.embeddings import HuggingFaceEmbeddings                    # (Old import) This line would import the embedding model from LangChain, but it's commented out.

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader  # Imports PDF and directory loaders from the community-supported LangChain module for loading PDF documents from directories.

from langchain_huggingface import HuggingFaceEmbeddings                    # Imports HuggingFaceEmbeddings to generate vector representations (embeddings) of text using Hugging Face models.


# ---------------- Function 1 ----------------
def load_pdf(data):                                                        # Defines a function named load_pdf that takes 'data' (a directory path) as input.
    loader=DirectoryLoader(data,                                           # Creates a DirectoryLoader object to load all PDF files from the given directory path.
                    glob="*.pdf",                                          # Specifies that only files with the .pdf extension should be loaded.
                    loader_cls=PyPDFLoader)                                # Uses PyPDFLoader as the class responsible for loading and parsing PDF files.
    
    documents=loader.load()                                                # Loads all the PDF documents from the specified directory and returns them as a list of document objects.
    return documents                                                       # Returns the loaded documents to the caller.


# ---------------- Function 2 ----------------
def text_split(extracted_data):                                            # Defines a function named text_split that takes extracted_data (list of documents) as input.
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)  # Creates a text splitter that divides text into chunks of up to 500 characters with a 20-character overlap, preserving context between chunks.
    text_chunks=text_splitter.split_documents(extracted_data)              # Splits the given documents into smaller text chunks using the defined text splitter.
    
    return text_chunks                                                     # Returns the list of split text chunks.


# ---------------- Function 3 ----------------
def download_hugging_face_embeddings():                                    # Defines a function named download_hugging_face_embeddings with no input parameters.
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Loads a pre-trained sentence transformer model from Hugging Face that converts text into numerical embeddings.
    
    return embeddings                                                      # Returns the Hugging Face embedding model object.
