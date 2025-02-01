import streamlit as st
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AstraDB
from langchain_groq import ChatGroq

st.title("💨 Layanan Pengguna AI JetStream Smart Hairdryer")
# Load BM25 index from file
def load_bm25_index(input_file: str) -> BM25Okapi:
    """Memuat indeks BM25 dari file."""
    with open(input_file, "rb") as f:
        bm25 = pickle.load(f)
    return bm25

# Fungsi pencarian fusion
def fusion_retrieval(vstore: AstraDB, bm25: BM25Okapi, query: str, k: int = 5, alpha: float = 0.5) -> list:
    """Melakukan pencarian gabungan BM25 dan pencarian berbasis vektor."""
    epsilon = 1e-8
    vector_results = vstore.similarity_search_with_score(query, k=k * 2)
    vector_docs = [doc for doc, _ in vector_results]
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    bm25_scores = bm25.get_scores(query.split())
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores[:len(vector_scores)]  
    sorted_indices = np.argsort(combined_scores)[::-1]
    return [vector_docs[i] for i in sorted_indices[:k]]

# Load secrets dari Streamlit
groq_api_key = st.secrets["groq"]["api_key"]
groq_model_name = st.secrets["groq"]["model_name"]
llm = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name=groq_model_name)

# Model HuggingFace dan tokenizer
huggingface_model_name = st.secrets["huggingface"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
model = AutoModel.from_pretrained(huggingface_model_name)
embedding_model = HuggingFaceEmbeddings(model_name=huggingface_model_name)

data_source = st.radio("Pilih Sumber Data:", ("PDF", "CSV"))
api_endpoint = st.secrets["astras"]["api_endpoint"]
token = st.secrets["astras"]["token"]
collection_name = "synthetic_data" if data_source == "PDF" else "synthetic_data_csv"
vstore = AstraDB(
    embedding=embedding_model,
    collection_name=collection_name,
    api_endpoint=api_endpoint,
    token=token,
)

bm25_index_path = "bm25_index.pkl" if data_source == "PDF" else "bm25_index_csv.pkl"
bm25_index = load_bm25_index(bm25_index_path)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ketik pesan Anda...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    top_docs = fusion_retrieval(vstore, bm25_index, prompt, k=5, alpha=0.7)
    docs_content = [doc.page_content for doc in top_docs]
    
    system_prompt = """Anda adalah layanan pelanggan yang sangat berpengetahuan tentang produk JetStream smart hair dryer. JAWAB DALAM BAHASA INDONESIA.
    Gunakan dokumen yang diambil untuk memberikan jawaban yang ketat, akurat, dan relevan.
    Jika tidak ada informasi yang relevan ditemukan, katakan bahwa Anda tidak tahu daripada mengarang jawaban."""
    
    chat_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state["messages"][-5:]])
    context = "\n\n".join(docs_content) + "\n\n"
    final_prompt = f"{system_prompt}\n\nRiwayat Percakapan:\n{chat_history}\n\nPertanyaan Pengguna: {prompt}\n\nKonteks:\n{context}"
    
    response = llm.invoke(final_prompt)
    response = response.content
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
