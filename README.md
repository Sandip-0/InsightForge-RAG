````markdown
# InsightForge-RAG

## Context-Aware Document Retrieval and Q&A Assistant

InsightForge is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents or provide website URLs and ask questions about their content.

The application combines document retrieval, vector embeddings, conversation history, query rewriting, and Mistral AI to provide context-aware answers.

## Live Demo

https://insightforge-sandip.streamlit.app

## GitHub Repository

https://github.com/Sandip-0/InsightForge-RAG

---

## Features

- Upload PDF documents
- Load content from website URLs
- Automatic document chunking
- Mistral AI embeddings
- Chroma vector database
- Semantic similarity search
- Maximum Marginal Relevance (MMR) retrieval
- Query rewriting for follow-up questions
- Conversation history
- Context-aware question answering
- Mistral Small LLM
- Streamlit web interface
- Markdown-formatted responses
- Session-based document storage
- Clear chat history
- Reset knowledge base and conversation

---

## How It Works

```text
                    User
                     |
                     v
             Streamlit Interface
                     |
          +----------+----------+
          |                     |
          v                     v
      PDF Upload            Website URL
          |                     |
          v                     v
    PyPDFLoader            WebBaseLoader
          |                     |
          +----------+----------+
                     |
                     v
              Text Splitting
                     |
                     v
       RecursiveCharacterTextSplitter
                     |
                     v
             Mistral Embeddings
                     |
                     v
                ChromaDB
                     |
                     v
              User Question
                     |
                     v
              Query Rewriting
                     |
                     v
             Vector Retrieval
                     |
                     v
             Relevant Context
                     |
          +----------+----------+
          |                     |
          v                     v
   Conversation History    Retrieved Context
          |                     |
          +----------+----------+
                     |
                     v
                 Mistral LLM
                     |
                     v
                   Answer
````

---

## RAG Pipeline

### 1. Document Loading

PDF files are loaded using `PyPDFLoader`.

Website content is loaded using `WebBaseLoader`.

### 2. Text Splitting

Large documents are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

### 3. Embeddings

Each document chunk is converted into a vector representation using `MistralAIEmbeddings`.

### 4. Vector Database

The embeddings are stored in `Chroma`, which retrieves document chunks that are semantically relevant to the user's question.

### 5. Query Rewriting

Before retrieval, the user's question is rewritten into a standalone search query.

For example:

```text
User: What is Word2Vec?

User: How does it work?
```

The second question can be rewritten as:

```text
How does Word2Vec work?
```

This helps the retriever understand follow-up questions.

### 6. Retrieval

The rewritten query is used to retrieve relevant document chunks from Chroma.

The application uses MMR retrieval to improve the diversity of retrieved results.

### 7. Context-Aware Generation

The final prompt contains:

```text
Conversation History
        +
Retrieved Context
        +
Current Question
```

The Mistral LLM then generates the final answer using the retrieved information.

---

## Technology Stack

| Technology                     | Purpose                         |
| ------------------------------ | ------------------------------- |
| Python                         | Programming language            |
| Streamlit                      | Web application interface       |
| LangChain                      | RAG orchestration               |
| Mistral AI                     | LLM and embeddings              |
| ChromaDB                       | Vector database                 |
| PyPDFLoader                    | PDF document loading            |
| WebBaseLoader                  | Website content loading         |
| RecursiveCharacterTextSplitter | Document chunking               |
| python-dotenv                  | Environment variable management |

---

## Project Structure

```text
InsightForge-RAG/
│
├── app.py
├── main.py
├── create_database.py
├── requirements.txt
├── .gitignore
│
├── document loders/
│   └── ...
│
├── retrivers/
│   └── ...
│
└── vector store/
    └── ...
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sandip-0/InsightForge-RAG.git
cd InsightForge-RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

Do not commit the `.env` file to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.DS_Store
```

---

## Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## How to Use

### Step 1 — Upload a Document

Open the application and use the sidebar to upload a PDF.

```text
Upload PDF
     ↓
Process PDF
     ↓
Document chunks
     ↓
Embeddings
     ↓
ChromaDB
```

### Step 2 — Add a Website

Alternatively, provide a website URL:

```text
https://example.com
```

Then click **Process URL**.

### Step 3 — Ask Questions

After adding a source, ask questions about the content.

Example:

```text
What is Word2Vec?
```

Then ask a follow-up:

```text
How does it work?
```

The query-rewriting component uses the conversation history to resolve the reference.

---

## Conversation History

InsightForge maintains conversation history during the current Streamlit session.

Example:

```text
User: What is Word2Vec?

AI: Word2Vec is a framework for generating word embeddings.

User: How does it work?

AI: Word2Vec uses CBOW and Skip-Gram models...
```

The history helps the system understand contextual follow-up questions.

---

## Query Rewriting

Query rewriting is an important part of the retrieval pipeline.

Without query rewriting:

```text
How does it work?
```

may be difficult for the vector retriever to understand because the query does not explicitly identify what `it` refers to.

With query rewriting:

```text
How does it work?
        ↓
Conversation History
        ↓
Query Rewriter
        ↓
How does Word2Vec work?
        ↓
Chroma Retrieval
```

This improves retrieval for conversational queries.

---

## Retrieval Configuration

The retriever uses Maximum Marginal Relevance (MMR):

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

### Parameters

* `k=4` — returns the top 4 relevant chunks
* `fetch_k=10` — initially considers 10 candidate chunks
* `lambda_mult=0.5` — balances relevance and diversity

---

## Session Management

The Streamlit application uses `st.session_state` to maintain:

```text
vector_store
chat_history
sources
```

The knowledge base and conversation are maintained for the current application session.

### Clear Chat History

Removes the conversation while keeping the loaded documents.

### Reset All

Removes:

* Conversation history
* Loaded sources
* Current vector store

and starts a fresh session.

---

## Example Use Cases

InsightForge can be used for:

* PDF question answering
* Research paper analysis
* Technical documentation Q&A
* Study material analysis
* Website content analysis
* Knowledge-base assistants
* Document-based AI assistants
* Conversational document search

---

## Limitations

* Answers depend on the quality of the retrieved context.
* The application is primarily designed for document-grounded question answering.
* Session-based data is not intended as permanent storage.
* Very large documents may require optimization for production use.
* LLM responses can still contain errors if the retrieved context is insufficient or ambiguous.

---

## Future Improvements

Possible improvements include:

* Persistent vector database
* PDF page-number citations
* Source citations in answers
* Multi-user authentication
* Persistent chat history
* Streaming LLM responses
* Hybrid keyword + semantic search
* Cross-encoder reranking
* Document metadata filtering
* Multiple knowledge bases
* Chat export functionality
* Docker deployment
* Cloud-based vector database
* Production monitoring and evaluation

---

## Learning Outcomes

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Vector embeddings
* Semantic search
* Vector databases
* Document chunking
* LLM prompting
* Query rewriting
* Conversational memory
* Context-aware question answering
* Streamlit application development
* LangChain integration
* Mistral AI integration

---

## Author

### Sandip Adak

B.Tech Computer Science & Engineering

Interested in:

* Data Science
* Machine Learning
* Generative AI
* Retrieval-Augmented Generation
* Natural Language Processing

---

## Project Links

**Live Demo:**
[https://insightforge-sandip.streamlit.app](https://insightforge-sandip.streamlit.app)

**GitHub:**
[https://github.com/Sandip-0/InsightForge-RAG](https://github.com/Sandip-0/InsightForge-RAG)

---

## License

This project is available for educational and personal use.

```
```
