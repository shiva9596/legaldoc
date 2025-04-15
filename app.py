import os
import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from cohere import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = Client(api_key=cohere_api_key)

# --- Functions ---

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text, len(reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
        return text, "N/A"
    else:
        return "", 0

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

def build_vectorstore(chunks):
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    return FAISS.from_documents(chunks, embeddings)

def generate_answer(context, question):
    response = cohere_client.chat(
        message=question,
        documents=[{"text": context}],
        model="command-r"
    )
    return response.text

# --- UI ---

st.set_page_config(page_title="Legal Document Q&A Assistant", layout="centered")
st.image("https://i.ibb.co/hVPCy6k/legal-header.png", use_container_width=True)
st.title("📄 Legal Document Q&A Assistant")

uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file:
    text, page_count = extract_text(uploaded_file)

    if not text.strip():
        st.warning("❌ No readable text found in the document.")
        st.stop()

    st.info(f"📄 Pages: {page_count if isinstance(page_count, int) else 'Unknown'}")

    chunks = chunk_text(text)
    st.info(f"🔍 Chunks Created: {len(chunks)}")

    vectorstore = build_vectorstore(chunks)

    # Suggested questions dropdown
    suggested_questions = [
        "What is the purpose of this document?",
        "What are the main clauses?",
        "Are there any liabilities mentioned?",
        "What parties are involved?",
        "What are the obligations and responsibilities?",
        "What is the validity period or effective date?",
        "Is there any termination clause?",
        "Are there any confidentiality or NDA clauses?",
        "What are the penalties or legal consequences?",
        "Are there dispute resolution procedures?"
    ]

    selected_question = st.selectbox("💡 Suggested Questions", [""] + suggested_questions)
    manual_question = st.text_input("Or type your own question")

    final_question = manual_question if manual_question else selected_question

    if st.button("🧠 Get Answer") and final_question:
        docs = vectorstore.similarity_search(final_question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = generate_answer(context, final_question)
        st.success("✅ Answer:")
        st.write(answer)
