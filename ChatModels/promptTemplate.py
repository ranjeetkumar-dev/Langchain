from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    # temperature=0         
)

template="write a {tone} email to {company} expressing intrest in the {position},mentioning {skill} as a key strength.keep it to 80 lines max and your name is skye "

prompt_templete=ChatPromptTemplate.from_template(template)

# print(prompt_templete)
prompt=prompt_templete.invoke({
    "tone":"energetic",
   "company": "samsung",
   "position":"backend developer",
   "skill":"Ai"
   
    
})

# print(prompt)
responce=llm.invoke(prompt)
print(responce.content)
