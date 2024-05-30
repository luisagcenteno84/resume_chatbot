import chainlit as cl
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough


def load_vector():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", max_batch_size=100)
    
    loader = PyPDFLoader(file_path="src/resume.pdf")
    data = loader.load()

    text_splitters = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=50)
    all_splits = text_splitters.split_documents(data)
    
    vector_store = FAISS.from_documents(all_splits, embeddings)

    print("FAISS vector store created:"+str(vector_store))

    return vector_store

def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

store = {}


@cl.on_chat_start
def on_start():
    vector_store = load_vector()
    retriever = vector_store.as_retriever()

    model = ChatGoogleGenerativeAI(model="models/gemini-pro", temperature=0.2)
    
    ### Contextualize question ###
    template = """
                    You are a CareerBot, a comprehensive, interactive resource for exploring Luis Gonzalez's background, skills, and expertise. 
                    Be polite and provide answers based on the provided context only and do not make up any data. 
                    
                    Use only the provided data and not prior knowledge.

                    Follow exactly these 3 steps:
                    1. Read the context below 
                    2. Answer the question using ONLY the provided context below. If the question cannot be answered based on the context below, politely decline to answer and say that you are only allowed to answer questions about Luis Gonzalez's career path, the places where he worked and the technologies he used, as well as the things he may be capable of building. 
                    3. Make sure to nicely format the output so it is easy to read on a small screen.

                    If you don't know the answer, just say you don't know. 
                    Do NOT try to make up an answer.
                    If the question is not related to the information about Luis Gonzalez, 
                    politely respond that you are tuned to only answer questions about Luis Gonzalez's experience, education, training and his aspirations. 
                    Use as much detail as possible when responding but keep your answer to up to 100 words.
                    At the end ask if the user would like to have more information or what else they would like to know about Luis Gonzalez.
                    
                    Context:
                    {context}

                    Question:
                    {question}
                """

    context_prompt = ChatPromptTemplate.from_template(template)

    
    rag_chain = (
        {"context": retriever,"question": RunnablePassthrough()}
        | context_prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("rag_chain",rag_chain)





@cl.on_message
async def on_message(message: cl.Message):

    rag_chain = cl.user_session.get("rag_chain")


    response = rag_chain.invoke(message.content)

    answer = cl.Message(response)

    await answer.send()
    


