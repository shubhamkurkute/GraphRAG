from langchain_ollama import ChatOllama
from document_chunking import split_n_chunk
from graph_db import create_graph, vector_store_db
from rag_chain import create_rag_chain

llm = ChatOllama(
    model="llava:7b",
    temperature=0.3,
    num_predict=256
)

chunks = split_n_chunk()
vector_store = vector_store_db(documents=chunks)
create_graph(documents=chunks, llm=llm)
response = create_rag_chain(vector_store, llm)
print(response)
