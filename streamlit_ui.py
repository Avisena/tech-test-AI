import streamlit as st
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import AstraDB
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Access secret configuration directly from Streamlit secrets
groq_api_key = st.secrets["groq"]["api_key"]
groq_model_name = st.secrets["groq"]["model_name"]
llm = ChatGroq(temperature=0.6, groq_api_key=groq_api_key, model_name=groq_model_name)

# HuggingFace model and tokenizer
huggingface_model_name = st.secrets["huggingface"]["model_name"]
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
model = AutoModel.from_pretrained(huggingface_model_name)

# Create HuggingFaceEmbeddings instance
embedding_model = HuggingFaceEmbeddings(model_name=huggingface_model_name)

# AstraDB configuration
api_endpoint = st.secrets["astras"]["api_endpoint"]
token = st.secrets["astras"]["token"]
vstore = AstraDB(
    embedding=embedding_model,
    collection_name="synthetic_data",
    api_endpoint=api_endpoint,
    token=token,
)

# Initialize Conversational Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state["chat_history"]:
    memory.save_context({"input": msg["user"]}, {"output": msg["assistant"]})

# Define system prompt
system_prompt = """You are a knowledgeable customer service for product JetStream smart hair dryer. Use the retrieved documents to provide strict, accurate, and relevant answers. If no relevant information is found, say you don't know rather than making up facts."""

# Create a custom prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}\n\nContext:\n{context}"),  # Embed context inside user input
])

# Create Conversational RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
)

# Streamlit UI
st.title("JetStream Smart Hairdryer AI Customer Support")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")
if prompt:
    pre_invoke_prompt = f"""
    this is the chat history:
    {st.session_state["messages"]}

    this is the question: {prompt}

    You are a knowledgeable customer service for product JetStream smart hair dryer. Based on question and chat history, create ONLY ONE sentence of what the query needed to vector search the information
    """
    pre_invoke = llm.invoke(pre_invoke_prompt)
    pre_invoke = pre_invoke.content
    retriever = vstore.as_retriever()
    print(retriever.get_relevant_documents(pre_invoke))
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response using RAG with memory
    response = qa_chain.run({
        "question": pre_invoke,
        "chat_history": memory.load_memory_variables({}),
    })    
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.session_state["chat_history"].append({"user": prompt, "assistant": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
