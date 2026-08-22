# from langchain_community.document_loaders import TextLoader
# data=TextLoader("document loders/notes.txt")
# # print(data)
# # print(data.load()[0].page_content)
# print(len(data.load()))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1,
)
data=TextLoader("document loders/notes.txt")
docs=data.load()
chunks=splitter.split_documents(docs)

for i in chunks:
    print(i.page_content)
    print("---------------------------------------------------")