from langchain_ollama import ChatOllama
from document_chunking import split_n_chunk
from graph_db import create_graph, vector_store_db
from rag_chain import create_rag_chain

llm = ChatOllama(
    model="gemma3:12b", 
)

chunks = split_n_chunk()
vector_store = vector_store_db(documents=chunks)
graph = create_graph(documents=chunks, llm=llm)
response = create_rag_chain(vector_store, llm, graph)
print(response)
