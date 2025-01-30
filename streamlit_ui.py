import streamlit as st
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings

tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en')

# Create HuggingFaceEmbeddings instance for Langchain
embedding_model = HuggingFaceEmbeddings(
    model_name='jinaai/jina-embeddings-v2-base-en', 
)

# Streamlit UI
st.title("Chatbot UI with Streamlit")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Placeholder for AI response (replace with your logic)
    response = "Hello! This is a placeholder response."
    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
