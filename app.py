import chainlit as cl
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

chat = GoogleGenerativeAI(model="models/gemini-pro", temperature=0.2)

@cl.on_chat_start
def on_start():
    return ""

@cl.on_message
async def on_message(message: cl.Message):

    prompt_template = PromptTemplate.from_template(
        ""
    )

    response = chat.invoke(
        [
            HumanMessage(content=message.content)
        ]
    )

    answer = cl.Message(content=response)

    await answer.send()
    