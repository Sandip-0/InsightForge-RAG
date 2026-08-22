## txt-----------------------------
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_community.document_loaders import TextLoader
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# data=TextLoader("document loders/notes.txt")
# docs=data.load()
# template=ChatPromptTemplate.from_messages([
#     ('system', "You are a AI that summarizes the text."),
#     ('human', "{data}")
# ])
# model =ChatMistralAI(model="mistral-small-2506")
# # prompt=template.format_messages(data=docs[0].page_content)
# prompt=template.format_messages(data=docs)
# result=model.invoke(prompt)
# print(result.content)










# # pdf --------------
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# data=PyPDFLoader("document loders/deeplearning-2.pdf")
# docs=data.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks=splitter.split_documents(docs)
# template=ChatPromptTemplate.from_messages([
#     ('system', "You are a AI that summarizes the text."),
#     ('human', "{data}")
# ])
# model =ChatMistralAI(model="mistral-small-2506")
# prompt=template.format_messages(data=chunks[0].page_content)
# result=model.invoke(prompt)
# print(result.content)







# # model create time
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

template=ChatPromptTemplate.from_messages([
    ('system', "You are a AI that summarizes the text."),
    ('human', "{data}")
])
model =ChatMistralAI(model="mistral-small-2506")


from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
embedding_model=MistralAIEmbeddings()
vector_store=Chroma(
    persist_directory="chroma_langchain_db",
    embedding_function=embedding_model
)
retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k':4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)
llm=ChatMistralAI(model="mistral-small-2506")
# prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ( "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            You may use the conversation history to understand
            references such as "it", "this", "that", etc.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """
        ),
        ("human",
           """Conversation History:{history}
                Context:{context}
                Question:{question}
            """
        )
    ]
)
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
        """Rewrite the user's question into a clear standalone
        search query for retrieving information from the document.

        Use the conversation history to understand words like
        "it", "this", and "that".

        Correct obvious spelling mistakes.
        Preserve important technical terms.

        Return ONLY the search query.
        Do not answer the question.
        """
    ),
    ("human",
        """Conversation History:{history}
            Question:{question}
        """
    )
])


def rewrite_query(question, history):

    rewrite_prompt = query_rewrite_prompt.invoke({
        "history": history,
        "question": question
    })

    response = llm.invoke(rewrite_prompt)

    return response.content.strip()
print("Rag system created.\npress 0 to exit.")
chat_history = []
while True:
    query=input("you : ")
    if query == "0":
        break
    history = "\n\n".join(
        [
            f"User: {item['question']}\nAI: {item['answer']}"
            for item in chat_history
        ]
    )
    search_query = rewrite_query(query, history)
    print(f"Search query: {search_query}")
    docs=retriever.invoke(search_query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )
    # Convert history into text
    final_prompt=prompt.invoke({
        "history": history,
        "context":context,
        "question":query
    })
    response=llm.invoke(final_prompt)
    # output
    print(f"\n AI : {response.content}")
    # Save current conversation
    chat_history.append({
        "question": query,
        "answer": response.content
    })
