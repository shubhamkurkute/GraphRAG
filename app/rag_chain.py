from langchain_graph_retriever import GraphRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_neo4j import GraphCypherQAChain

def create_rag_chain(vector_store, llm, graph):

    vector_docs = GraphRetriever(store=vector_store)
    graph_traversal = GraphCypherQAChain.from_llm(llm=llm, graph=graph, allow_dangerous_requests=True)

    def format_docs(docs):
        if not docs:
            return "No context found."
        return "\n\n".join(doc.page_content for doc in docs)

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert AI assistant. Use both the vector-based and graph-based information to answer accurately.

    --- Vector Context ---
    {vector_context}

    --- Graph Context ---
    {graph_context}

    --- Question ---
    {question}

    --- Answer ---
    """)

    # Create chain
    chain = (
        {
            "vector_context": vector_docs | format_docs,
            "graph_context": lambda x: graph_traversal.invoke(x),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke("Where is the wafer located?")
    return response


