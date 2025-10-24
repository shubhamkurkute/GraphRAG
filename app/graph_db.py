from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

graph_tranformer = LLMGraphTransformer()

URI = "neo4j://127.0.0.1:7687"
PASSWORD = "12345678"
USERNAME = "neo4j"

graph = Neo4jGraph(
    url=URI,
    username=USERNAME,
    password=PASSWORD
)

print(graph)
# def create_n_store(chunks, llm):
#     llm_transformer = LLMGraphTransformer(llm=llm)
#     graph_documents = llm_transformer.to_graph_document(chunks)






