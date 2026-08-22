from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]
embedding_model = MistralAIEmbeddings(model="mistral-embed")  # Replace with your actual embedding model
vector_store=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

similar_docs = vector_store.similarity_search("what is data analysis?",k=2)
for r in similar_docs:
    print(r.page_content)

retriever = vector_store.as_retriever()
docs=retriever.invoke("what is data analysis?",k=2)

for r in docs:
    print(r.page_content)
