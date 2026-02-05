import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# -------------------------------------------------
# Environment & Client Setup
# -------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not set. Configure Streamlit Secrets.")
    st.stop()

client = OpenAI()

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# -------------------------------------------------
# Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
        .card {
            background-color: #161b22;
            padding: 1.2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .answer-box {
            background-color: #0d1117;
            padding: 1.2rem;
            border-radius: 8px;
            border-left: 4px solid #1f4ed8;
            white-space: pre-wrap;
        }
        .citation {
            color: #9da5b4;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("⚖️ Legal Document AI Assistant")
st.caption("Chat with legal documents using OpenAI + FAISS RAG")

# -------------------------------------------------
# Helpers: Text Extraction
# -------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
    return pages

def extract_text_from_docx(uploaded_file):
    document = docx.Document(uploaded_file)
    full_text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(1, full_text)]

# -------------------------------------------------
# Build Vector Store
# -------------------------------------------------
def build_vector_store(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    texts = []
    metadatas = []

    for page_num, text in pages:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"page": page_num})

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )

# -------------------------------------------------
# Query LLM (Streaming + Citations)
# -------------------------------------------------
def query_llm(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(
        f"(Page {d.metadata['page']}) {d.page_content}"
        for d in docs
    )

    system_prompt = (
        "You are a legal assistant. "
        "Answer strictly from the provided context. "
        "If the answer is not present, say so clearly. "
        "Cite page numbers when relevant."
    )

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    answer = ""
    placeholder = st.empty()

    with client.responses.stream(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                answer += event.delta
                placeholder.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True,
                )

    return answer, docs

# -------------------------------------------------
# Sidebar: Upload & Preview
# -------------------------------------------------
with st.sidebar:
    st.subheader("📄 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
    )

    if uploaded_file and uploaded_file.name.lower().endswith(".pdf"):
        st.subheader("👀 Preview")
        st.pdf(uploaded_file)

# -------------------------------------------------
# Main App Logic
# -------------------------------------------------
if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        pages = extract_text_from_pdf(uploaded_file)
    else:
        pages = extract_text_from_docx(uploaded_file)

    if not pages:
        st.warning("No text found in document.")
        st.stop()

    if "vectorstore" not in st.session_state:
        with st.spinner("Indexing document..."):
            st.session_state.vectorstore = build_vector_store(pages)
            st.session_state.chat = []

    st.success("✅ Document indexed. Ask your questions.")

    # ---------------- Chat UI ----------------
    st.subheader("💬 Chat")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a legal question...")

    if user_question:
        st.session_state.chat.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("assistant"):
            answer, sources = query_llm(
                user_question,
                st.session_state.vectorstore,
            )

            st.markdown(
                "<div class='citation'><b>Sources:</b></div>",
                unsafe_allow_html=True,
            )
            for s in sources:
                st.markdown(
                    f"<div class='citation'>• Page {s.metadata['page']}</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.chat.append(
            {"role": "assistant", "content": answer}
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <center style="color:#8b949e;font-size:0.8rem;">
    Legal AI Assistant · OpenAI · Streamlit · FAISS
    </center>
    """,
    unsafe_allow_html=True,
)
