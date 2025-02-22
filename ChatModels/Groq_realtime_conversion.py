from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq LLM

llm = ChatGroq(
           model_name="llama-3.3-70b-versatile",
    temperature=0
)

chat_history=[]

system_message=SystemMessage("You are a math expert assistant.")

chat_history.append(system_message)

while(True):
    query=input("You:")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))
    result=llm.invoke(chat_history)
    responce=result.content
    chat_history.append(responce)
    print(f"AI:{responce}")


# print("____Message_history___")
# print(chat_history)



