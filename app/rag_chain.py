from langchain_graph_retriever import GraphRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def create_rag_chain(vector_store, llm):
    query = "Who is the founder and the name of company founded"
    graph_retrieval = GraphRetriever(store=vector_store)

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


