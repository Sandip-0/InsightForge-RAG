from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings=MistralAIEmbeddings()
vectorstore=Chroma.from_documents(docs,embeddings)
# retriever=vectorstore.as_retriever() // use similar search
retriever=vectorstore.as_retriever(search_type='mmr')
llm=ChatMistralAI()
multi_query_retriver=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)
query='What is gradient descent?'
docs=multi_query_retriver.invoke(query)
print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)