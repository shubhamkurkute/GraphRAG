from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

URI = "neo4j://127.0.0.1:7687"
PASSWORD = "12345678"
USERNAME = "neo4j"

graph = Neo4jGraph(
    url=URI,
    username=USERNAME,
    password=PASSWORD
)


def create_graph(documents, llm):
    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(documents=documents)
    store_graph_documents(graph_documents=graph_documents)


def store_graph_documents(graph_documents):
    """Store graph documents in Neo4j"""
    print("Storing graph data in Neo4j...")
    
    # Clear existing data (optional)
    graph.query("MATCH (n) DETACH DELETE n")
    
    for graph_doc in graph_documents:
        # Add nodes
        for node in graph_doc.nodes:
            node_type = node.type.replace(" ", "_")  # Sanitize for Neo4j
            graph.query(
                f"MERGE (n:{node_type} {{id: $id}})",
                {"id": node.id}
            )
        
        # Add relationships
        for rel in graph_doc.relationships:
            rel_type = rel.type.replace(" ", "_")
            graph.query(
                f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                """,
                {
                    "source_id": rel.source.id,
                    "target_id": rel.target.id
                }
            )
    
    print("Graph data stored successfully")


def vector_store_db(documents):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore.from_documents(documents=documents,embedding=embedding)
    return vector_store




