# 📄 PDF Reader Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Powered-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-58%20Passing-success?style=for-the-badge)

An **AI-powered PDF chatbot** that lets users upload any PDF, ask natural-language questions about it, and even talk to it — with spoken questions and spoken answers. The system extracts and semantically indexes PDF text, retrieves the most relevant sections for each query, and uses an LLM to generate accurate, context-aware answers.

---

## ✨ Features

- 📄 **PDF Q&A (RAG)** — Upload a PDF and ask questions in natural language, answered using retrieval-augmented generation over your document
- 🔍 **Semantic Search** — FAISS vector index + sentence-transformer embeddings (`all-MiniLM-L6-v2`) retrieve the most relevant chunks for every question
- 🎙️ **Voice Input** — Speak your question instead of typing; transcribed via Groq's `whisper-large-v3-turbo`
- 🔊 **Spoken Answers (TTS)** — Bot replies can be read aloud, with a choice of six voices (Hannah, Autumn, Diana, Austin, Daniel, Troy)
- ✅ **Well-Tested** — 58 automated tests covering document processing, retrieval, RAG, and voice services

---

## 📸 Demo

```
User uploads: "project_report.pdf"

User: "What is the main objective of this project?"
Bot:  "The main objective is to build a real-time data pipeline that..."

User: 🎙️ (spoken) "Summarise the conclusion section."
Bot:  🔊 "The conclusion highlights three key findings: ..."
```

---

## ⚙️ How It Works

1. **PDF Upload** — User uploads a PDF file through the Streamlit interface
2. **Text Extraction & Chunking** — Readable text is extracted and split into overlapping chunks for better context
3. **Embedding & Indexing** — Chunks are embedded with a sentence-transformer model and indexed in FAISS
4. **Question Input** — User types a question, or asks it by voice (transcribed via Groq Whisper)
5. **Semantic Retrieval** — The most relevant chunks are retrieved for the question
6. **LLM Answer Generation** — Retrieved context is passed to a Groq-hosted LLM, which generates a precise answer
7. **Spoken Answer (optional)** — The answer can be converted to speech and played back

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| Streamlit | Web-based user interface |
| pypdf | PDF text extraction |
| LangChain / langchain-groq | LLM chaining and retrieval pipeline |
| Groq API | LLM inference, Whisper transcription, and TTS |
| sentence-transformers | Text embeddings for semantic search |
| FAISS | Vector store for document retrieval |
| pytest | Automated testing (58 tests) |
| Ruff | Linting |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/arunkumararavindhakshan05-sudo/Pdf-Reader-Chatbot.git
cd Pdf-Reader-Chatbot

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file (or set a Streamlit secret) with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, upload a PDF, and start asking questions — by typing or by voice.

---

## 📁 Project Structure

```
Pdf-Reader-Chatbot/
│
├── app.py                          # Main Streamlit application
├── src/
│   ├── document_processor.py       # PDF text extraction & chunking
│   ├── retriever.py                # FAISS-based semantic retrieval
│   ├── rag_service.py              # RAG pipeline (Groq LLM)
│   └── voice_service.py            # Voice transcription & TTS (Groq)
├── tests/                          # 58 automated tests (pytest)
├── requirements.txt
└── README.md
```

---

## 🧪 Testing & Quality

```bash
# Run tests
pytest

# Run linting
ruff check .
```

---

## 🚀 Live Demo

Deployed on **Streamlit Cloud** — try it live: _[add your Streamlit Cloud app URL here]_

---

## 🔮 Future Improvements

- [ ] Support multiple PDF uploads at once
- [ ] Add chat history / memory across sessions
- [ ] Support Word (.docx) and Excel (.xlsx) files

---

## 👤 Author

**Arunkumar Aravindhakshan**
🔗 [LinkedIn](https://linkedin.com/in/arunkumar-aravindhakshan) | [GitHub](https://github.com/arunkumararavindhakshan05-sudo)