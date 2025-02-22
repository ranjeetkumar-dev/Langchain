from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
# from langchain_groq import ChatGroq
from langchain_ollama.llms import OllamaLLM



from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
# llm = ChatGroq(
#     model_name="llama-3.3-70b-versatile",
#     # temperature=0
# )

llm = OllamaLLM(model="llama3.1",temperature=0,token=10)

# template="you are a {animal} expert. give me the {count} facts about it. "

# prompt_templete=ChatPromptTemplate.from_template(template)
# prompt=prompt_templete.invoke({
#     "animal":"dog",
#     "count":"5"
# })
# # print(prompt)
# responce=llm.invoke(prompt)
# print(responce.content)

template=[("system","you are a {subject} expert."),
                 ("human","tell me {count} facts about it.")]

prompt_templete=ChatPromptTemplate.from_messages(template)
# prompt=prompt_templete.invoke({
#     "subject":"computer",
#     "count":"5"
    
# })
# print(prompt)
# res=llm.invoke(prompt)
# print(res.content)
chain=prompt_templete | llm

res=chain.invoke({"subject":"laptop","count":5})
# print(res.content)
print(res)


