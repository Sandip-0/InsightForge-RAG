import streamlit as st
import tempfile
import os
import html
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Assistant", page_icon="✨", layout="wide")

# Custom styling for dark modern chat theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }
    
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    
    .user-bubble-container {
        display: flex;
        justify-content: flex-end;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        width: 100%;
    }
    
    .user-bubble {
        background-color: #1e3a8a;
        color: #ffffff;
        padding: 12px 20px;
        border-radius: 20px 20px 4px 20px;
        max-width: 75%;
        font-size: 15px;
        line-height: 1.55;
        word-wrap: break-word;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
    }
    
    .chat-divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Mistral LLM
@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")

llm = get_llm()

# Create a fresh, in-memory Chroma vector store for each session
def get_fresh_vectorstore():
    embedding_model = MistralAIEmbeddings()
    # In-memory Chroma: clean and resets completely on reload
    return Chroma(embedding_function=embedding_model)

# Session State Initialization (resets completely on browser reload)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_fresh_vectorstore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sources" not in st.session_state:
    st.session_state.sources = []

# RAG prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
        """You are a helpful, accurate, and professional AI assistant.
        Use the provided context to answer the user's question.
        You may use the conversation history to understand context, pronouns, and references.
        If the relevant answer cannot be found in the context, politely state:
        "I could not find information about that in the provided documents."
        Format your answers clearly using markdown headings, bullet points, and bold text where appropriate.
        """
    ),
    ("human",
       """Conversation History:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}"""
    )
])

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
        """Given the conversation history and a user question, formulate a standalone search query to retrieve relevant documents.
        Resolve any pronouns (it, this, that, he, she, etc.) using the conversation history.
        Return ONLY the search query keywords. Do not include quotes, preamble, or answers.
        """
    ),
    ("human",
        """Conversation History:\n{history}\n\nQuestion:\n{question}"""
    )
])

# Process PDF upload into session vector store
def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        if chunks:
            st.session_state.vector_store.add_documents(chunks)
        return len(chunks)
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Process website link into session vector store
def process_url(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        header_template={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    if chunks:
        st.session_state.vector_store.add_documents(chunks)
    return len(chunks)

def rewrite_query(question, history):
    if not history.strip():
        return question
    rewrite_prompt = query_rewrite_prompt.invoke({
        "history": history,
        "question": question
    })
    response = llm.invoke(rewrite_prompt)
    clean_query = response.content.strip().strip('"').strip("'")
    return clean_query if clean_query else question

# --- SIDEBAR (Knowledge Base & Controls) ---
with st.sidebar:
    st.title("Knowledge Base 📚")
    
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded_file and st.button("Process PDF", use_container_width=True):
        with st.spinner("Processing PDF..."):
            try:
                num_chunks = process_uploaded_file(uploaded_file)
                st.session_state.sources.append(f"{uploaded_file.name} ({num_chunks} chunks)")
                st.success(f"Added {num_chunks} chunks from PDF!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            
    st.subheader("Add Website")
    url_input = st.text_input("Enter URL", placeholder="https://example.com", label_visibility="collapsed")
    if url_input and st.button("Process URL", use_container_width=True):
        with st.spinner("Fetching and indexing website..."):
            try:
                num_chunks = process_url(url_input)
                st.session_state.sources.append(f"{url_input} ({num_chunks} chunks)")
                st.success(f"Indexed {num_chunks} chunks from URL!")
            except Exception as e:
                st.error(f"Error loading URL: {e}")
                
    st.markdown("<hr style='border: 0.5px solid #30363d; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.subheader("Active Sources")
    if st.session_state.sources:
        for source in st.session_state.sources:
            st.caption(f"📄 {source}")
    else:
        st.caption("No sources loaded. Please upload a PDF or enter a website URL above.")
        
    st.markdown("<hr style='border: 0.5px solid #30363d; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.subheader("Session Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    if st.button("🧹 Reset All (Delete Data & History)", use_container_width=True):
        st.session_state.vector_store = get_fresh_vectorstore()
        st.session_state.chat_history = []
        st.session_state.sources = []
        st.rerun()

# --- MAIN CHAT DISPLAY ---
st.title("AI Assistant ✨")

# 1. Display previous messages
for item in st.session_state.chat_history:
    user_escaped = html.escape(item["question"]).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="user-bubble-container">
            <div class="user-bubble">{user_escaped}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(item["answer"])
    st.markdown("<hr class='chat-divider'>", unsafe_allow_html=True)

# 2. Chat input
if query := st.chat_input("Ask anything..."):
    # Check if any sources have been added
    if not st.session_state.sources:
        st.warning("⚠️ Please upload a PDF or enter a Website URL in the left sidebar first!")
        st.stop()
        
    history = "\n\n".join(
        [
            f"User: {item['question']}\nAI: {item['answer']}"
            for item in st.session_state.chat_history
        ]
    )
    
    with st.spinner("Thinking..."):
        search_query = rewrite_query(query, history)
        
        # Retrieve relevant chunks from the current session's database
        docs = st.session_state.vector_store.similarity_search(search_query, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        final_prompt = prompt.invoke({
            "history": history,
            "context": context,
            "question": query
        })
        
        response = llm.invoke(final_prompt)
        answer_text = response.content
    
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer_text
    })
    st.rerun()