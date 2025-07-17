# from flask import Flask,render_template,jsonify,request
# from langchain.vectorstores import Pinecone
# from src.helper import download_hugging_face_embeddings
# import pinecone
# from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from src.prompt import *
# import os


# app=Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

# embeddings=download_hugging_face_embeddings()

# #

# from pinecone import Pinecone, ServerlessSpec, PodSpec # Make sure PodSpec is imported if you might use it

# # Initialize Pinecone
# # Replace with your actual API key and environment
# # Ensure your environment matches the region you plan to use if it's region-specific
# PINECONE_API_KEY = PINECONE_API_KEY
# PINECONE_API_ENV = PINECONE_API_ENV # e.g., "us-east-1-aws" or similar, check your dashboard

# pinecone = Pinecone(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_API_ENV
# )

# index_name = "medical-chatbot"

# # Check if the index exists
# # pinecone.list_indexes() returns a list of dictionaries, so we need to check names
# existing_indexes = pinecone.list_indexes()
# existing_index_names = [idx['name'] for idx in existing_indexes]

# if index_name not in existing_index_names:
#     print(f"Creating index '{index_name}'...")
#     pinecone.create_index(
#         name=index_name,
#         dimension=1536,  # Ensure this matches your embedding model's dimension
#         metric='cosine',
#         spec=ServerlessSpec(cloud='aws', region='us-east-1') # Or your free-tier supported region
#     )
#     print(f"Index '{index_name}' created successfully.")
# else:
#     print(f"Index '{index_name}' already exists. Connecting to it.")

# # Now you can connect to your index
# index = pinecone.Index(index_name)

# # You can also add a small check to ensure it's ready, especially after creation
# # This is more relevant for Pod-based indexes, but good practice
# # try:
# #     # For Serverless, it should be ready almost immediately after creation call returns
# #     # For Pod, you might need to loop and check index.describe_index_stats() until 'ready'
# #     index.describe_index_stats()
# #     print(f"Successfully connected to and can query index '{index_name}'.")
# # except Exception as e:
# #     print(f"Error connecting to index '{index_name}': {e}")

# print(f"Index object: {index}")

# index_name="medical-chatbot"
# #
# docsearch=Pinecone.from_existing_index(index_name,embeddings)

# PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
# chain_type_kwargs={"prompt":PROMPT}

# llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                   model_type="llama",
#                   config={'max_new_tokens':512,
#                           'temperature':0.8})

# from langchain.vectorstores import Pinecone

# # Wrap Pinecone index as LangChain VectorStore
# vectorstore = Pinecone(
#     index=index,
#     embedding_function=embeddings.embed_query,
#     text_key="text"
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# from langchain.chains import RetrievalQA

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,  # Must come from langchain.vectorstores.Pinecone
#     return_source_documents=True,  # ✅ fixed typo
#     chain_type_kwargs=chain_type_kwargs
# )


# @app.route("/")
# def index():
#     return render_template('chat.html')

# if __name__ == '__main__':
#     app.run(debug=True)






#---------------------------------------------------------------

from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from langchain_pinecone import Pinecone


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from src.helper import download_hugging_face_embeddings
from src.prompt import *
import os


app = Flask(__name__)
load_dotenv()

# Load credentials
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

index_name = "medical-chatbot"

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # make sure it matches your embedding dimension
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print("Index created.")
else:
    print(f"Index '{index_name}' already exists.")

# Connect to existing index
index = pc.Index(index_name)

# LangChain Pinecone wrapper
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="text"  # optional, remove if not using namespaces
)

# Build retriever & chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 512,
        'temperature': 0.8
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

###add
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"response": "Please provide a query."}), 400

    try:
        result = qa({"query": query})
        response = result["result"]
        return jsonify({"response": response})
    except Exception as e:
        print("Error during QA:", e)
        return jsonify({"response": "Sorry, something went wrong."}), 500


if __name__ == '__main__':
    app.run(debug=True)
