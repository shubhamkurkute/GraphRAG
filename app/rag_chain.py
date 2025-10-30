from langchain_graph_retriever import GraphRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_neo4j import GraphCypherQAChain

def create_rag_chain(vector_store, llm, graph):
    query = "What anomaly can be seen in the overall scenario"
    vector_docs = GraphRetriever(store=vector_store)
    graph_traversal = GraphCypherQAChain(graph_store=graph)
    graph_docs = graph_traversal.from_llm(llm=llm, cypher_prompt=query)
    print(graph_docs)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
        # Prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the provided context.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """)
    
    # Create chain
    chain = (
        {
            "context": graph_retrieval | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(query)
    return response


