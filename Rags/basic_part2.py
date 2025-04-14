import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama.llms import OllamaLLM

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up the retriever
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)

# Initialize the model
model = ChatGroq(model_name="llama-3.3-70b-versatile")
# model = OllamaLLM(model="llama3.1")

print("Type 'exit' to quit.")

while True:
    # Get user query
    query = input("Enter your query: ")
    if query.lower() == "exit":
        break

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(query)

    # Display the relevant results
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    # Combine input for the model
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide a rough answer based only on the provided documents. "
        "If the answer is not found in the documents, respond with 'I'm not sure'."
    )

    # Define messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model
    result = model.invoke(messages)

    # Display the response
    print("\n--- Generated Response ---")
    print(result.content)
