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

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()

st.set_page_config(
    page_title="InsightForge AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS (BLUE-GLASS THEME + TRANSPARENT BOTTOM FIX)
# ============================================================
st.markdown(
    """
    <style>
    /* 1. MAIN APP BACKGROUND - Radial Gradient */
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.08), transparent 30%),
            radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08), transparent 30%),
            #0b0f17 !important;
        background-attachment: fixed !important;
        color: #e6edf3;
    }
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 8rem;
    }
    #MainMenu, footer, header { visibility: hidden !important; background: transparent !important; }

    /* 2. SIDEBAR */
    section[data-testid="stSidebar"] {
        background: #080b12 !important;
        border-right: 1px solid #1a2230 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* 3. GEMINI WELCOME SCREEN */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 15vh;
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    .welcome-title {
        font-size: 3.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.04em;
    }
    .welcome-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }

    /* 4. CHAT BUBBLES */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 1.2rem 0 !important;
    }
    
    /* User Message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex;
        flex-direction: row-reverse;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        background: #1e293b !important;
        color: #f8fafc !important;
        padding: 0.85rem 1.4rem !important;
        border-radius: 20px 20px 4px 20px !important;
        max-width: 80% !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    }

    /* Assistant Message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
        background: transparent !important;
        color: #e2e8f0 !important;
        padding: 0 !important;
        max-width: 100% !important;
        line-height: 1.7;
    }
    
    /* Hide Avatars */
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }

    /* 5. THE BOTTOM CONTAINER FIX (REMOVES BLACK BOX) */
    [data-testid="stBottom"], 
    [data-testid="stBottom"] > div,
    .stBottomBlockContainer {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* 6. CHAT INPUT BAR - Blue-Glass */
    [data-testid="stChatInput"] {
        background: transparent !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stChatInput"] > div {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid #334155 !important;
        border-radius: 30px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4) !important;
        display: flex !important;
        flex-direction: row-reverse !important; /* Send button on left */
        align-items: center !important;
        padding: 4px 8px !important;
        transition: border 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        background: rgba(15, 23, 42, 0.95) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(96, 165, 250, 0.3) !important;
    }
    
    /* Transparent internal inputs */
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="input"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        background: transparent !important;
        font-size: 1rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    /* Source Cards */
    .source-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.85rem;
        transition: border 0.2s;
    }
    .source-card:hover {
        border-color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LLM & VECTOR STORE SETUP
# ============================================================
@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.1)

@st.cache_resource
def get_embedding_model():
    return MistralAIEmbeddings(model="mistral-embed")

llm = get_llm()
embedding_model = get_embedding_model()

def get_fresh_vectorstore():
    return Chroma(embedding_function=embedding_model)

# ============================================================
# SESSION STATE
# ============================================================
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_fresh_vectorstore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# ============================================================
# PROMPTS
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are InsightForge, an AI document assistant. Answer the user's question using the provided context. If the answer is not in the context, politely state that you cannot find it. Use Markdown for clarity."),
    ("human", "History:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}")
])

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question into a standalone search query based on history. Return ONLY the search keywords."),
    ("human", "History:\n{history}\n\nQuestion:\n{question}")
])

# ============================================================
# PROCESSORS
# ============================================================
def process_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_sig = (uploaded_file.name, len(file_bytes))

    if file_sig in st.session_state.processed_files:
        return 0, "PDF already processed."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        if not docs: return 0, "No readable pages found."
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        if not chunks: return 0, "Could not chunk PDF."
        
        st.session_state.vector_store.add_documents(chunks)
        st.session_state.processed_files.add(file_sig)
        return len(chunks), None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def process_url(url: str):
    loader = WebBaseLoader(
        web_paths=(url,),
        header_template={"User-Agent": "Mozilla/5.0"}
    )
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
    if chunks: st.session_state.vector_store.add_documents(chunks)
    return len(chunks)

def rewrite_query(question: str, history: str):
    if not history.strip(): return question
    try:
        res = llm.invoke(query_rewrite_prompt.invoke({"history": history, "question": question}))
        clean = res.content.strip().strip('"\'')
        return clean if clean else question
    except:
        return question

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("<h3 style='color:#E6F1FF; margin-bottom: 1.5rem;'>InsightForge-RAG 🔎</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file and st.button("Process PDF", use_container_width=True):
        with st.spinner("Indexing..."):
            chunks, err = process_uploaded_file(uploaded_file)
            if err: st.warning(err)
            else:
                st.session_state.sources.append({"name": uploaded_file.name, "type": "PDF", "chunks": chunks})
                st.success(f"Indexed {chunks} chunks!")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
    url_input = st.text_input("Add Website", placeholder="https://...", label_visibility="collapsed")
    if st.button("Process URL", use_container_width=True):
        if url_input:
            with st.spinner("Indexing..."):
                try:
                    chunks = process_url(url_input.strip())
                    if chunks > 0:
                        st.session_state.sources.append({"name": url_input.strip(), "type": "URL", "chunks": chunks})
                        st.success(f"Indexed {chunks} chunks!")
                except Exception as e: st.error(f"Failed: {e}")

    st.markdown("<hr style='border-color: #1e293b; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    st.markdown("<strong style='color: #cbd5e1;'>Active Sources</strong>", unsafe_allow_html=True)
    if st.session_state.sources:
        for s in st.session_state.sources:
            st.markdown(f"<div class='source-card'><b style='color:#e2e8f0;'>{s['type']}</b><br><span style='color:#94a3b8;'>{html.escape(s['name'])}</span><br><span style='color:#64748b; font-size: 0.75rem;'>{s['chunks']} chunks</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;'>No sources loaded yet.</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1e293b; margin: 2rem 0;'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True): st.session_state.chat_history = []; st.rerun()
    if st.button("🧹 Reset Data", use_container_width=True): 
        st.session_state.vector_store = get_fresh_vectorstore()
        st.session_state.chat_history = []
        st.session_state.sources = []
        st.session_state.processed_files = set()
        st.rerun()

# ============================================================
# MAIN UI
# ============================================================
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">Hello, there</div>
            <div class="welcome-subtitle">How can InsightForge help you explore your documents today?</div>
        </div>
        """, unsafe_allow_html=True
    )

for item in st.session_state.chat_history:
    with st.chat_message("user"): st.markdown(item["question"])
    with st.chat_message("assistant"): st.markdown(item["answer"])

if query := st.chat_input("Ask anything about your documents..."):
    if not st.session_state.sources:
        st.warning("Please upload a PDF or enter a URL first!")
        st.stop()

    history = "\n".join([f"User: {i['question']}\nAI: {i['answer']}" for i in st.session_state.chat_history])
    
    with st.spinner("Thinking..."):
        search_query = rewrite_query(query, history)
        docs = st.session_state.vector_store.similarity_search(search_query, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        
        response = llm.invoke(prompt.invoke({"history": history, "context": context, "question": query}))
        
    st.session_state.chat_history.append({"question": query, "answer": response.content})
    st.rerun()