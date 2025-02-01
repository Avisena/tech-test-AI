import streamlit as st
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AstraDB
from langchain_groq import ChatGroq
import time

# Set UI title
st.title("\U0001F4A8 Layanan Pengguna AI JetStream Smart Hairdryer")

# Function to load BM25 index from file
def load_bm25_index(input_file: str) -> BM25Okapi:
    """Loads BM25 index from a pickle file."""
    try:
        with open(input_file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading BM25 index: {e}")
        return None

# Hybrid retrieval function combining BM25 and vector search
def fusion_retrieval(vstore: AstraDB, bm25: BM25Okapi, query: str, k: int = 5, alpha: float = 0.5) -> list:
    """Performs a hybrid search combining BM25 and vector-based search results."""
    epsilon = 1e-8  # Small constant to avoid division by zero

    # Step 1: Retrieve vector-based results from AstraDB
    vector_results = vstore.similarity_search_with_score(query, k=k * 2)
    vector_docs = [doc for doc, _ in vector_results]
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    # Step 2: Retrieve BM25 scores
    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # Step 3: Weighted fusion of scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores[:len(vector_scores)]  
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Step 4: Return top k documents
    return [vector_docs[i] for i in sorted_indices[:k]]

# Load secrets from Streamlit configuration
groq_api_key = st.secrets["groq"]["api_key"]
groq_model_name = st.secrets["groq"]["model_name"]
llm = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name=groq_model_name)

# Load Hugging Face model and tokenizer
huggingface_model_name = st.secrets["huggingface"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
model = AutoModel.from_pretrained(huggingface_model_name)
embedding_model = HuggingFaceEmbeddings(model_name=huggingface_model_name)

# UI selection for data source
data_source = st.radio("Pilih Sumber Data:", ("PDF", "CSV"))

# Configure AstraDB connection
api_endpoint = st.secrets["astras"]["api_endpoint"]
token = st.secrets["astras"]["token"]
collection_name = "synthetic_data" if data_source == "PDF" else "synthetic_data_csv"
vstore = AstraDB(
    embedding=embedding_model,
    collection_name=collection_name,
    api_endpoint=api_endpoint,
    token=token,
)

# Load BM25 index
bm25_index_path = "bm25_index.pkl" if data_source == "PDF" else "bm25_index_csv.pkl"
bm25_index = load_bm25_index(bm25_index_path)
if bm25_index is None:
    st.stop()

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
prompt = st.chat_input("Ketik pesan Anda...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant documents using hybrid search
    top_docs = fusion_retrieval(vstore, bm25_index, prompt, k=5, alpha=0.5)
    docs_content = [doc.page_content for doc in top_docs]
    
    # Define system instructions
    system_prompt = """Anda adalah layanan pelanggan yang sangat berpengetahuan tentang produk JetStream smart hair dryer. JAWAB DALAM BAHASA INDONESIA.
    Gunakan dokumen yang diambil untuk memberikan jawaban yang ketat, akurat, dan relevan.
    Jika tidak ada informasi yang relevan ditemukan, katakan bahwa Anda tidak tahu daripada mengarang jawaban."""
    
    # Prepare chat history
    chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"][-5:]])
    context = "\n\n".join(docs_content) + "\n\n"
    final_prompt = f"{system_prompt}\n\nRiwayat Percakapan:\n{chat_history}\n\nPertanyaan Pengguna: {prompt}\n\nKonteks:\n{context}"
    
    # Generate AI response with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(final_prompt)
            response = response.content
            break
        except Exception as e:
            st.warning(f"Percobaan {attempt + 1} gagal: {e}")
            time.sleep(2)  # Wait before retrying
    else:
        response = "Maaf, terjadi kesalahan dalam menghasilkan jawaban. Silakan coba lagi nanti."
    
    # Append AI response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
