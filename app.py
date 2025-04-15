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

# ---- UI ----
st.set_page_config(page_title="Legal Doc AI Assistant", layout="centered")
st.title("📄 Legal Document AI Assistant")

cohere_key = os.getenv("COHERE_API_KEY")
if not cohere_key:
    st.error("Cohere API key not found. Please add it in Hugging Face Secrets.")
    st.stop()
os.environ["COHERE_API_KEY"] = cohere_key

uploaded_pdf = st.file_uploader("📎 Upload a legal PDF", type=["pdf"])

# Suggested questions
st.markdown("### 💡 Suggested Questions")
suggested = [
    "What are the key obligations mentioned in the document?",
    "Is there any mention of termination clauses?",
    "What rights does the tenant have?",
    "Does the contract mention penalties or liabilities?"
]
for q in suggested:
    st.markdown(f"- {q}")

question = st.text_input("❓ Ask your legal question here")
submit = st.button("🚀 Submit Question")

if uploaded_pdf and submit and question:
    with st.spinner("Processing your document and question..."):

        # Extract PDF text
        reader = PdfReader(uploaded_pdf)
        raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        page_count = len(reader.pages)

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.create_documents([raw_text])
        chunk_count = len(docs)

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

        llm = Cohere(model="command-r-plus", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        response = qa_chain.run(question)

        # Display results
        st.success("📬 AI-generated Answer:")
        st.markdown(f"**{response.strip()}**")
        st.info(f"📄 Processed {page_count} pages | 🔍 Created {chunk_count} text chunks")

elif uploaded_pdf and not question and submit:
    st.warning("⚠️ Please enter a question to proceed.")
