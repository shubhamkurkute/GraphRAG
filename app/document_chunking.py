from langchain_community.document_loaders import PyMuPDFLoader, JSONLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import jq


def split_n_chunk():
    chunks = []
    # loader = PyMuPDFLoader("sample.pdf")
    loader = JSONLoader(file_path="analysis.json",jq_schema=".frame_analyses[].response")

    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
    chunks = splitter.split_documents(document[:1])

    return chunks
