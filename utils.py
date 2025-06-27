import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader

def load_contracts_from_folder(folder_path="contract_templates"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                metadata = {"source": filename.split(".")[0]}
                docs.append(Document(page_content=text, metadata=metadata))
    return docs

def build_vector_store(docs, persist_path="embeddings"):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding)
    vectordb.save_local(persist_path)
    return vectordb

def load_docs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            fpath = os.path.join(folder_path, filename)
            try:
                documents.extend(TextLoader(fpath).load())
            except Exception as e:
                print(f"❌ Error loading {fpath}: {e}")
    return documents
