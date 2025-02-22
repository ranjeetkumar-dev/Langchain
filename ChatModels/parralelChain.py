from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

from langchain_ollama.llms import OllamaLLM

from dotenv import load_dotenv

load_dotenv()


# model creation

model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    # temperature=0
)
model2 = OllamaLLM(model="llama3.1")

# task 1 #initial task
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a movie critic"),
        ("human", "provide a brif summary  of the movie {movie_name}."),
    ]
)


# def plot analysis    #parallel task 1
def plot_analysis(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a movie critic."),
            ("human", "Analyse the plot:{plot}.what are its weaknesses and strength?."),
        ]
    )
    return plot_template.format_prompt(plot=plot)


# def character analysis  #parallel task 2


def character_analysis(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a movie critic."),
            (
                "human",
                "analyse the characters:{characters} what are their strength and weaknesses?.",
            ),
        ]
    )

    return character_template.format_prompt(characters=characters)


# combine analyse final   #task final


def combine_verdict(plot_analysis, character_analysis):
    return (
        f"plot analysis:\n{plot_analysis}\n \ncharacter analysis\n:{character_analysis}"
    )


# paralel chain 1
plot_branch_chain = (
    RunnableLambda(lambda x: plot_analysis(x)) | model | StrOutputParser()
)
# parallel chain 2
character_branch_chain = (
    RunnableLambda(lambda x: character_analysis(x)) | model2 | StrOutputParser()
)

# create the combine chain

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(
        branches={"plot": plot_branch_chain, "characters": character_branch_chain}
    )
    | RunnableLambda(
        lambda x: combine_verdict(x["branches"]["plot"], x["branches"]["characters"])
    )
)

# run the chain
result = chain.invoke({"movie_name": "Inception"})
print(result)
