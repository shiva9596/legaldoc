import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import docx2txt
import tempfile

# Load environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App UI
st.set_page_config(page_title="Legal Document AI", layout="centered")
st.title("📄 Legal Document AI Assistant")

# File uploader
uploaded_file = st.file_uploader("📎 Upload a legal document (.pdf or .docx)", type=["pdf", "docx"])
question = st.text_input("💬 Ask a legal question")
submit = st.button("🚀 Submit")

# Extract text from uploaded file
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    return None

# RAG logic
if uploaded_file and question and submit:
    with st.spinner("Processing..."):
        raw_text = extract_text(uploaded_file)
        if not raw_text:
            st.error("❌ Could not extract text from the document.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.create_documents([raw_text])

            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful legal assistant. Use the context below to answer the user's question.

If the answer is not in the document, say: "I'm not sure based on the document."

Context:
{context}

Question:
{question}

Answer:
"""
            )

            llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.3)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )

            answer = qa_chain.run(question)
            st.success("📬 Answer:")
            st.markdown(answer)

elif uploaded_file and submit and not question:
    st.warning("⚠️ Please enter a question.")
