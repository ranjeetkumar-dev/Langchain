from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
[]
# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

messages=[
        SystemMessage("You are social media content expert."),
        HumanMessage("give a short trick to create engaging posts on instagram."),
        AIMessage("2+2 is 4")


]

ans=llm.invoke(messages)
# ans=llm.invoke("who is the prisident of india")
print(ans.content)