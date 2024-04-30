import chainlit as cl
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import StrOutputParser


@cl.on_chat_start
def on_start():
    model = GoogleGenerativeAI(model="models/gemini-pro", temperature=0.2, streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an experienced leader working on a retailer"
            ),
            ("human","{question}")
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable",runnable)


@cl.on_message
async def on_message(message: cl.Message):

    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question":message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await message.send()
    