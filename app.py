import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx2txt
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

st.set_page_config(page_title="Legal Document AI", layout="centered")
st.title("📄 Legal Document AI Assistant")

# File uploader (PDF or Word)
uploaded_file = st.file_uploader("📎 Upload a legal document (.pdf or .docx)", type=["pdf", "docx"])

# Suggested & custom questions
suggested_questions = [
    "📌 What are the key clauses mentioned in this document?",
    "📌 Are there any termination conditions?",
    "📌 What are the penalties or liabilities?",
    "📌 Is there a confidentiality agreement?",
    "📌 Who are the involved parties?"
]

st.markdown("### 💡 Choose a suggested question or type your own:")
col1, col2 = st.columns(2)
with col1:
    dropdown_question = st.selectbox("Suggested Questions", ["Select..."] + suggested_questions)
with col2:
    custom_question = st.text_input("Or write your own question")

submit = st.button("🚀 Submit")

# Text extraction from uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    return None

# Decide final question
final_question = None
if custom_question.strip():
    final_question = custom_question.strip()
elif dropdown_question != "Select...":
    final_question = dropdown_question

# RAG workflow
if uploaded_file and final_question and submit:
    with st.spinner("Processing document..."):
        raw_text = extract_text(uploaded_file)

        if not raw_text:
            st.error("❌ Unable to extract text from the document.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.create_documents([raw_text])

            embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are a helpful AI legal assistant. Use the context below to answer the user's question.

If the answer is not present, respond: "I'm not sure based on the document."

Context:
{context}

Question:
{question}

Answer:
"""
            )

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )

            answer = qa_chain.run(final_question)
            st.success("📬 Answer:")
            st.markdown(answer)

elif uploaded_file and submit and not final_question:
    st.warning("⚠️ Please enter or select a question.")
