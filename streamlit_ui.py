import streamlit as st
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AstraDB
from langchain_groq import ChatGroq

# Load BM25 index from file
def load_bm25_index(input_file: str) -> BM25Okapi:
    """Load a BM25 index from a file."""
    with open(input_file, "rb") as f:
        bm25 = pickle.load(f)
    return bm25

# Fusion retrieval function
def fusion_retrieval(vstore: AstraDB, bm25: BM25Okapi, query: str, k: int = 5, alpha: float = 0.5) -> list:
    """Perform fusion retrieval combining BM25 and vector-based search."""
    epsilon = 1e-8  # Small constant to avoid division by zero

    # Step 1: Retrieve documents from AstraDB using vector search
    vector_results = vstore.similarity_search_with_score(query, k=k * 2)  # Retrieve more for better fusion
    vector_docs = [doc for doc, _ in vector_results]
    vector_scores = np.array([score for _, score in vector_results])

    # Normalize vector scores (convert distance to similarity)
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Normalize BM25 scores
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # Step 3: Combine scores using weighted sum
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores[:len(vector_scores)]  

    # Step 4: Rank documents based on combined scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Step 5: Return top k documents
    return [vector_docs[i] for i in sorted_indices[:k]]

# Load secrets from Streamlit
groq_api_key = st.secrets["groq"]["api_key"]
groq_model_name = st.secrets["groq"]["model_name"]
llm = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name=groq_model_name)

# HuggingFace model and tokenizer
huggingface_model_name = st.secrets["huggingface"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
model = AutoModel.from_pretrained(huggingface_model_name)

# Create HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name=huggingface_model_name)

# UI for selecting data source
data_source = st.radio("Select Data Source:", ("PDF", "CSV"))

# AstraDB configuration
api_endpoint = st.secrets["astras"]["api_endpoint"]
token = st.secrets["astras"]["token"]
collection_name = "synthetic_data" if data_source == "PDF" else "synthetic_data_csv"
vstore = AstraDB(
    embedding=embedding_model,
    collection_name=collection_name,
    api_endpoint=api_endpoint,
    token=token,
)

# Load BM25 index based on the selected data source
bm25_index_path = "bm25_index.pkl" if data_source == "PDF" else "bm25_index_csv.pkl"
bm25_index = load_bm25_index(bm25_index_path)

# Initialize memory container
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")
if prompt:
    # Add user input to the chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve top documents using BM25 + Vector fusion
    top_docs = fusion_retrieval(vstore, bm25_index, prompt, k=5, alpha=0.7)
    docs_content = [doc.page_content for doc in top_docs]

    # System prompt for controlling the assistant's behavior
    system_prompt = """You are a knowledgeable customer service for product JetStream smart hair dryer. 
    Use the retrieved documents to provide strict, accurate, and relevant answers. 
    If no relevant information is found, say you don't know rather than making up facts."""

    # Combine the user query and the retrieved context with the system prompt
    context = "\n\n".join(docs_content) + "\n\n"

    # Add the system prompt to the user query
    final_prompt = f"{system_prompt}\n\nUser Query: {prompt}\n\nContext:\n{context}"

    # AI response generation (using Groq model)
    response = llm.invoke(final_prompt)  # Pass the final_prompt
    response = response.content
    # Add assistant's response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
