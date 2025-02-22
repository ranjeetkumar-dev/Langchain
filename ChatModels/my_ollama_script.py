
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.1",temperature=2,token=10)
ans=llm.invoke("hi")
print(ans)