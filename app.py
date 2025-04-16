import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to query OpenAI (or any LLM)
def query_llm(question, context):
    prompt = f"""You are a legal assistant AI. Answer the question based on the following document content:

Document:
\"\"\"{context}\"\"\"

Question: {question}
Answer:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit UI
st.set_page_config(page_title="Legal Document Q&A Assistant", page_icon="📄")
st.title("📄 Legal Document Q&A Assistant")
st.markdown("Upload a **PDF** or **DOCX** file")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx"])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1]
    
    try:
        if file_ext == "pdf":
            full_text = extract_text_from_pdf(uploaded_file)
        elif file_ext == "docx":
            full_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"✅ Document processed successfully ({len(full_text.split())} words)")

    # Suggested questions
    suggested_questions = [
        "What is the main purpose of this document?",
        "Who are the parties involved?",
        "What are the key dates mentioned?",
        "What legal obligations are specified?"
    ]

    st.markdown("### 💡 Suggested Questions")
    selected_question = st.selectbox("Choose a question or ask your own:", suggested_questions)
    custom_question = st.text_input("Or ask your own:", value=selected_question)

    if st.button("🔍 Submit Question"):
        with st.spinner("Generating answer..."):
            answer = query_llm(custom_question, full_text)
        st.markdown("### 🧠 Answer")
        st.write(answer)
