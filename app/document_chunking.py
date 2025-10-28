from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



def split_n_chunk():
    chunks = []
    loader = PyMuPDFLoader("sample.pdf")
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
    chunks = splitter.split_documents(document)

    return chunks
