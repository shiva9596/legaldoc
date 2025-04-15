import streamlit as st
import os
from uuid import uuid4
from PyPDF2 import PdfReader
import docx2txt
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatCohere
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load .env if running locally (optional, harmless on Streamlit Cloud)
load_dotenv()

# Load Cohere API key
cohere_api_key = os.getenv("COHERE_API_KEY")

st.set_page_config(page_title="Legal Document Q&A", layout="wide")
st.title("📄 Legal Document AI Assistant")
st.markdown("Upload a legal document and ask questions. The assistant will retrieve relevant answers using Cohere + RAG.")

# Helper: Extract text from PDF or DOCX
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return ""

# Upload Document
uploaded_file = st.file_uploader("📤 Upload a legal PDF or DOCX file", type=["pdf", "docx"])
if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Extract and split text
    text = extract_text(uploaded_file)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    st.info(f"📄 Total Chunks: {len(chunks)} | 🧩 Total Pages (est.): {len(chunks)//3}")

    # Display first 2 chunks optionally
    with st.expander("🔍 Preview Sample Chunks"):
        st.write(chunks[:2])

    # Embed + VectorStore
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # RAG pipeline setup
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatCohere(cohere_api_key=cohere_api_key, model="command-r"),
        retriever=vectorstore.as_retriever()
    )

    # Suggested Questions
    st.subheader("💬 Ask a Question")
    suggestions = [
        "What is the contract duration?",
        "Who are the parties involved?",
        "Are there any termination clauses?",
        "What is the risk assessment?",
        "What are the financial obligations?"
    ]
    question = st.selectbox("Choose a question or type your own:", suggestions)
    custom_question = st.text_input("Or ask your own question:")
    final_question = custom_question if custom_question else question

    # Submit button
    if st.button("🧠 Get Answer"):
        if final_question.strip() == "":
            st.warning("⚠️ Please enter a valid question.")
        else:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(final_question)
                st.success("✅ Answer:")
                st.write(answer)
