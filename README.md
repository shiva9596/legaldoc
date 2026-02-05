
# 📄 Legal Document AI Assistant

An AI-powered Streamlit application that allows users to upload legal documents (PDF or Word), ask legal questions, and receive intelligent, context-aware answers using Cohere embeddings and a GPT-based language model via LangChain.

---

## 🚀 Features

- 🗂 Uploads: Supports **PDF (.pdf)** and **Word (.docx)** legal documents
- 🧠 AI-Powered Q&A: Uses **RAG (Retrieval-Augmented Generation)** for accurate responses
- 🔍 Vector Search: FAISS vector store for fast document chunk retrieval
- 💬 Question Input: Choose from suggested legal questions or type your own
- 🧾 Secure: API keys loaded via `.env` file
- 🧠 Powered by:
  - [LangChain](https://www.langchain.com/)
  - [OpenAI / GPT-3.5 / GPT-4](https://platform.openai.com/)

---

## 🛠 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/shiva9596/legaldoc.git
cd legaldoc
```

### 2. Create and populate `.env`

```env
OPENAI_API_KEY=your-OPENAI-api-key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app locally

```bash
streamlit run app.py
```

## 📎 Supported File Formats

- PDF (`.pdf`)
- Word Document (`.docx`)

---

## 💡 Example Questions

- What are the key clauses mentioned?
- Are there any termination conditions?
- What are the penalties or liabilities?
- Is there a confidentiality agreement?

---

## 📄 License

MIT License

## Project Live Demo
The project is live! You can check it out at the following link below:
```bash
https://p9qikwkggvsjf7jdgdqtvc.streamlit.app/
```
