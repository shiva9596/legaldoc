# ⚖️ Legal Document AI Assistant

An AI-powered **Streamlit application** that enables users to upload legal documents (PDF or Word), ask natural-language legal questions, and receive **accurate, context-aware answers** using **Retrieval-Augmented Generation (RAG)** with OpenAI and FAISS.

This project is designed for **legal document analysis**, ensuring responses are strictly grounded in the uploaded content with **clear source citations**.

---

## 🚀 Key Features

- 📄 **Document Upload**
  - Supports **PDF (.pdf)** and **Word (.docx)** files
- 💬 **Chat-Style Q&A Interface**
  - Ask follow-up questions conversationally
- 🧠 **Retrieval-Augmented Generation (RAG)**
  - Intelligent chunking and semantic search for precise answers
- 🔍 **Vector Search with FAISS**
  - Fast similarity search across document chunks
- 🧾 **Source Citations**
  - Answers reference relevant document pages
- ⚡ **Streaming Responses**
  - Answers stream in real time for a better user experience
- 👀 **In-App PDF Preview**
  - Preview uploaded PDFs directly in the sidebar
- 🎨 **Custom Theming**
  - Professional dark UI using Streamlit theming
- ☁️ **Streamlit Cloud Ready**
  - Secure API key handling via Streamlit Secrets

---

## 🧠 Tech Stack

- **Frontend / App Framework**
  - Streamlit
- **LLM & Embeddings**
  - OpenAI (GPT-4o-mini, text-embedding-3-small)
- **RAG Framework**
  - LangChain
- **Vector Store**
  - FAISS
- **Document Parsing**
  - PyPDF2, python-docx

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shiva9596/legaldoc.git
cd legaldoc
2️⃣ Configure Environment Variables
Option A: Streamlit Cloud (Recommended)
Add the following in Streamlit → App Settings → Secrets:

OPENAI_API_KEY = "your-openai-api-key"
Option B: Local .env File
OPENAI_API_KEY=your-openai-api-key
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the App Locally
streamlit run app.py
📎 Supported File Formats
PDF (.pdf)

Word Document (.docx)

💡 Example Questions
What is the purpose of this agreement?

Who are the parties involved?

Are there any termination or renewal clauses?

What liabilities or penalties are mentioned?

Is there a confidentiality or NDA clause?

What deadlines or obligations are specified?

🌐 Live Demo
The application is deployed on Streamlit Cloud:

🔗 Live App:
https://p9qikwkggvsjf7jdgdqtvc.streamlit.app/

📄 License
This project is licensed under the MIT License.

📌 Notes
The assistant answers strictly based on the uploaded document

If the information is not present, the model clearly states that

Suitable for contract review, legal analysis, and compliance checks

Easily extensible to multi-document comparison, clause highlighting, and exports
