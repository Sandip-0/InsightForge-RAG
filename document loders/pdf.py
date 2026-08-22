# from langchain_community.document_loaders import PyPDFLoader
# data=PyPDFLoader("document loders/GRU.pdf")
# docs=data.load()
# # print(len(data.load()))
# # print(docs)
# print(docs[14].page_content)


# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

# splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
# data=PyPDFLoader("document loders/GRU.pdf")
# docs=data.load()
# chunks=splitter.split_documents(docs)
# print(chunks[0].page_content)


# that is good
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
data=PyPDFLoader("document loders/GRU.pdf")
docs=data.load()
chunks=splitter.split_documents(docs)
print(chunks[0].page_content)