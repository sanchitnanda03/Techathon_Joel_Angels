import os
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from utils import load_contracts_from_folder, build_vector_store
import language_tool_python
import textstat
import json

# Open and load the full JSON file
with open("contract_clause_keyword.json", "r") as f:
    required_keywords = json.load(f)

 
def calculate_actqm(contract_text, contract_type, penalty_factor=0.5):
    # Clause Coverage Score (CCS)
    must_have_keywords = required_keywords.get(contract_type, [])
    found_keywords = [kw for kw in must_have_keywords if kw.lower() in contract_text.lower()]
    ccs = len(found_keywords) / len(must_have_keywords) if must_have_keywords else 0
 
    # Formality Score (FS) - less strict penalty
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(contract_text)
    grammar_issues = len(matches)
    sentence_count = max(textstat.sentence_count(contract_text), 1)
    fs = 1 - penalty_factor * (grammar_issues / sentence_count)
    fs = max(fs, 0)  # ensure FS not below zero
 
    # ACTQM Score
    actqm = ((2 * ccs + fs) / 3) * 100
    return {
        "CCS": round(ccs, 2),
        "FS": round(fs, 2),
        "ACTQM": round(actqm, 2),
        "Keywords Found": found_keywords,
        "Total Required": len(must_have_keywords),
        "Grammar Issues": grammar_issues,
        "Sentences": sentence_count
    }

# Setup and configuration
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📄 Contract Template Generator", layout="centered")
st.title("🤖 Contract Template Generator using Azure GPT-4o-mini + RAG")

# Setup embedding index path
EMBED_PATH = "embeddings"
INDEX_FILE = os.path.join(EMBED_PATH, "index.faiss")

# Load vector store
if not os.path.exists(INDEX_FILE):
    st.info("🔄 First-time setup: creating vector store from templates...")
    docs = load_contracts_from_folder("contract_templates")
    vectordb = build_vector_store(docs, EMBED_PATH)
else:
    vectordb = FAISS.load_local(
        EMBED_PATH,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    azure_endpoint="https://openaiqc.gep.com/techathon/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
    openai_api_version="2025-01-01-preview",
    deployment_name="gpt-4o-mini",
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

if "history" not in st.session_state:
    st.session_state.history = []

# UI Section
st.subheader("🔎 Request a Contract Template")
contract_type = st.text_input("Enter contract type (e.g., NDA, MSA, Employment Agreement)")

@traceable(name="generate_contract_template_interaction")
def generate_contract_template(query):
    return qa_chain.run(query)

if st.button("🚀 Generate Template") and contract_type:
    query = f"Create a detailed contract template for {contract_type}. Make sure it is formal, general-purpose, and does not include any party names."
    st.session_state.history.append(("user", query))
    result = generate_contract_template(query)
    st.session_state.history.append(("ai", result))

    st.subheader("📑 Generated Contract Template")
    st.text_area("Generated Template", result, height=400)

    filename = f"{contract_type.lower().replace(' ', '_')}_template.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result)

    with open(filename, "rb") as f:
        st.download_button("📥 Download Template as Text File", f, file_name=filename, mime="text/plain")
    
    st.subheader("📊 ACTQM: Automated Quality Evaluation")
    metrics = calculate_actqm(result, contract_type.strip())
    st.markdown(f"""
        - **Clause Coverage Score (CCS)**: `{metrics['CCS']} ({len(metrics['Keywords Found'])}/{metrics['Total Required']} keywords found)`
        - **Formality Score (FS)**: `{metrics['FS']} ({metrics['Grammar Issues']} issues in {metrics['Sentences']} sentences)`
        - **✅ ACTQM**: `{metrics['ACTQM']}`
        """)

# Feedback and history
for i, (role, msg) in enumerate(st.session_state.history):
    if role == "ai":
        feedback_key = f"feedback_{i}"
        feedback = st.radio(
            "Was this response helpful?",
            ("👍 Yes", "👎 No"),
            key=feedback_key
        )
        if feedback == "👍 Yes":
            st.success("✅ Thanks for your feedback!")
        elif feedback == "👎 No":
            st.warning("❗ We appreciate your feedback and will improve.")
            client_feedback = st.text_input("Please provide the changes you require:", key=f"feedback_text_{i}")
            if st.button("Submit changes", key=f"submit_changes_{i}") and client_feedback:
                modified_query = f"Make the changes to {msg} based on the feedback: {client_feedback}"
                modified_result = generate_contract_template(modified_query)
                st.session_state.history.append(("ai", modified_result))
                st.subheader("📑 Modified Contract Template")
                st.text_area("Modified Template", modified_result, height=400)
                filename = f"{contract_type.lower().replace(' ', '_')}_template.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(modified_result)
                with open(filename, "rb") as f:
                    st.download_button("📥 Download Template as Text File", f, file_name=filename, mime="text/plain")
                
                st.subheader("📊 ACTQM: Automated Quality Evaluation")
                metrics = calculate_actqm(modified_result, contract_type.strip())
                st.markdown(f"""
                    - **Clause Coverage Score (CCS)**: `{metrics['CCS']} ({len(metrics['Keywords Found'])}/{metrics['Total Required']} keywords found)`
                    - **Formality Score (FS)**: `{metrics['FS']} ({metrics['Grammar Issues']} issues in {metrics['Sentences']} sentences)`
                    - **✅ ACTQM**: `{metrics['ACTQM']}`
                    """)
