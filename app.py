import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import cohere

# Load environment variables
load_dotenv()
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

# -------- Helper Functions --------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

def query_llm(question, context):
    prompt = f"""You are a helpful legal assistant. Answer the question based on the document content below.

Document:
\"\"\"{context}\"\"\"

Question: {question}
Answer:"""
    response = cohere_client.chat(
        message=prompt,
        model="command-r",
        temperature=0.3
    )
    return response.text

# -------- Streamlit UI --------
st.set_page_config(page_title="Legal Document Q&A Assistant", page_icon="📄")
st.title("📄 Legal Document Q&A Assistant")
st.markdown("Upload a **PDF** or **DOCX** file to ask questions about its content.")

uploaded_file = st.file_uploader("📤 Upload your document", type=["pdf", "docx"])

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

    if full_text:
        st.success(f"✅ Document processed successfully! ({len(full_text.split())} words)")

        st.markdown("### 💡 Suggested Questions")
        suggested_questions = [
            "What is the main purpose of this document?",
            "Who are the parties involved?",
            "Are there any important deadlines mentioned?",
            "What legal obligations are specified?",
            "Summarize the key takeaways from this document."
        ]

        selected_question = st.selectbox("Choose a suggested question:", suggested_questions)
        user_question = st.text_input("Or ask your own question:", value=selected_question)

        if st.button("🔍 Submit"):
            with st.spinner("Generating an answer..."):
                answer = query_llm(user_question, full_text)
            st.markdown("### 🧠 Answer")
            st.write(answer)
    else:
        st.warning("The uploaded document appears to be empty.")
