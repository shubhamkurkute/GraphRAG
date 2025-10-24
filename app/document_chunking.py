from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_n_chunk():
    chunks = []
    loader = PyMuPDFLoader("Essential-GraphRAG.pdf")
    datas = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for data in datas:
        chunk = splitter.split_text(data.page_content)
        chunks.append(chunk)

    return chunks