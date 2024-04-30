# leader_chatbot

```mermaid
    flowchart LR
        A(Chainlit Message) --> |Enter Message| B(Langchain Prompt)
        B --> C(Gemini 1.5)
        C --> A 

```