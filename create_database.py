# load to pdf
# split to chunks
# create the embeddings
# store into chrome db


from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings

loder=PyPDFLoader("document loders/deeplearning-2.pdf")
docs=loder.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks=splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(model="mistral-embed")  # Replace with your actual embedding model
vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
