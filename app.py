import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from openai import OpenAI

# -------------------------------------------------
# Environment & Client Setup
# -------------------------------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY is not set. Please configure Streamlit secrets.")
    st.stop()

client = OpenAI()

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Legal Document Q&A",
    page_icon="📄",
    layout="centered",
)

# -------------------------------------------------
# Custom Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #9da5b4;
            margin-bottom: 1.5rem;
        }
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
        .footer {
            color: #8b949e;
            font-size: 0.8rem;
            text-align: center;
            margin-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="main-title">📄 Legal Document Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a legal document and ask precise, document-grounded questions</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(uploaded_file):
    document = docx.Document(uploaded_file)
    return "\n".join(
        [para.text for para in document.paragraphs if para.text.strip()]
    )

def query_llm(question, context):
    system_prompt = (
        "You are a legal assistant. "
        "Answer questions strictly using the provided document. "
        "If the answer is not present, say so explicitly."
    )

    user_prompt = f"""
Document:
\"\"\"{context}\"\"\"

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a PDF or DOCX file",
    type=["pdf", "docx"],
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()

    try:
        if ext == "pdf":
            full_text = extract_text_from_pdf(uploaded_file)
        elif ext == "docx":
            full_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    if not full_text:
        st.warning("The uploaded document appears to be empty.")
        st.stop()

    st.success(f"✅ Document loaded successfully ({len(full_text.split())} words)")

    # ---------------- Question Section ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("❓ Ask a Question")

    suggested_questions = [
        "What is the main purpose of this document?",
        "Who are the parties involved?",
        "Are there any important deadlines mentioned?",
        "What legal obligations are specified?",
        "Summarize the key takeaways from this document.",
    ]

    selected_question = st.selectbox(
        "Suggested questions",
        suggested_questions,
    )

    user_question = st.text_input(
        "Or ask your own question",
        value=selected_question,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Answer Section ----------------
    if st.button("🔍 Submit", use_container_width=True):
        with st.spinner("Analyzing document..."):
            answer = query_llm(user_question, full_text)

        st.markdown("### 🧠 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with OpenAI · Streamlit · Python<br>
        Legal Q&A Assistant
    </div>
    """,
    unsafe_allow_html=True,
)
