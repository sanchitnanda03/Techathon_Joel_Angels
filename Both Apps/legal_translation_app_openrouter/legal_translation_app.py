
# legal_translation_app.py

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- 1. SETUP DIRECTORIES ---
DOC_DIR = "./reference_docs"
VECTOR_DIR = "./faiss_index"

# --- 2. LOAD AND EMBED REFERENCE DOCUMENTS ---
def prepare_vector_store():
    if os.path.exists(VECTOR_DIR):
        return FAISS.load_local(VECTOR_DIR, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in os.listdir(DOC_DIR):
        if file.endswith(".txt"):
            file_path = os.path.join(DOC_DIR, file)
            loader = TextLoader(file_path=file_path, encoding="utf-8")
            text_doc = loader.load()
            split_docs = splitter.split_documents(text_doc)
            docs.extend(split_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_DIR)
    return vectorstore

# --- 3. LOAD LLAMA 3 VIA OPENROUTER ---
def load_llm(openrouter_key):
    return ChatOpenAI(
        temperature=0.1,
        model_name="meta-llama/llama-3-8b-instruct:nitrosocke",
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )

# --- 4. DEFINE TRANSLATION CHAIN ---
def build_translation_chain(api_key):
    retriever = prepare_vector_store().as_retriever(search_kwargs={"k": 3})
    llm = load_llm(api_key)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return chain

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Legal Contract Translator", layout="centered")
st.title("📄 NDA Legal Translator")

st.markdown("""
Paste an NDA contract in Language **Y** and this tool will translate it into Language **X**, aligning with legal wording from your reference documents.
""")

openrouter_key = st.text_input("🔐 Enter your OpenRouter API Key:", type="password")
user_input = st.text_area("📤 Paste NDA contract in Language Y:", height=300)

if st.button("Translate to Language X"):
    if not openrouter_key.strip():
        st.warning("Please enter your OpenRouter API key.")
    elif not user_input.strip():
        st.warning("Please paste the contract text.")
    else:
        with st.spinner("Translating and aligning legal terms..."):
            chain = build_translation_chain(openrouter_key)
            prompt = f"Translate the following NDA contract to Language X using legal terminology consistent with the provided NDA documents. Ensure legal fidelity and clause equivalence.\n\n{user_input}"
            result = chain.run(prompt)
        st.success("Translation complete:")
        st.text_area("Translated NDA in Language X:", result, height=300)
