import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import subprocess
import sys

# Automatically install langchain-community if not already installed
try:
    import langchain_community
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-community"])
    import langchain_community

# Streamlit Config
st.set_page_config(page_title="Legal Doc AI Assistant", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #fce4ec);
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.8em;
        margin-top: 0.5em;
        margin-bottom: 0.3em;
    }

    .stFileUploader, .stTextInput, .stSelectbox, .stButton {
        background-color: rgba(255, 255, 255, 0.75);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    .stMarkdown {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    .stInfo {
        background-color: rgba(230, 244, 255, 0.7);
        color: #0277bd;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📄 Legal Document AI Assistant")

# API Key Handling
cohere_key = os.getenv("COHERE_API_KEY")
if not cohere_key:
    st.error("Cohere API key not found. Please add it in Hugging Face Secrets.")
    st.stop()
os.environ["COHERE_API_KEY"] = cohere_key

# Upload PDF
uploaded_pdf = st.file_uploader("📎 Upload a legal PDF", type=["pdf"])

# Dropdown Suggested Questions
suggested = [
    "Select a question...",
    "What are the key obligations mentioned in the document?",
    "Is there any mention of termination clauses?",
    "What rights does the tenant have?",
    "Does the contract mention penalties or liabilities?"
]

question = st.selectbox("❓ Ask your legal question here", options=suggested)

# Submit Button
submit = st.button("🚀 Submit Question")

# Run model if conditions met
if uploaded_pdf and submit and question != suggested[0]:
    with st.spinner("Processing your document and question..."):

        # Extract PDF text
        reader = PdfReader(uploaded_pdf)
        raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        page_count = len(reader.pages)

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.create_documents([raw_text])

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a legal assistant AI. Use the following context to answer the user's legal question concisely and clearly.

Context:
{context}

Question:
{question}

Answer:"""
        )

        # LLM
        llm = Cohere(model="command-r-plus", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        # Get Answer
        try:
            response = qa_chain.run(question)
            st.success("📬 AI-generated Answer:")
            st.markdown(f"**{response.strip()}**")
            st.info(f"📄 Processed {page_count} page(s) from uploaded document.")
        except Exception as e:
            st.error(f"❌ Something went wrong while generating the answer.\n\n{e}")

elif uploaded_pdf and submit and question == suggested[0]:
    st.warning("⚠️ Please select a valid question from the dropdown.")

