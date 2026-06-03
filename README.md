# 📄 PDF Reader Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Powered-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

An **AI-powered PDF chatbot** that allows users to upload any PDF document and ask natural language questions about its content. The system extracts text from the PDF and uses a Large Language Model (LLM) to generate accurate, context-aware answers.

---

## 📸 Demo

```
User uploads: "project_report.pdf"

User: "What is the main objective of this project?"
Bot:  "The main objective is to build a real-time data pipeline that..."

User: "Summarise the conclusion section."
Bot:  "The conclusion highlights three key findings: ..."
```

---

## ⚙️ How It Works

1. **PDF Upload** — User uploads a PDF file through the interface
2. **Text Extraction** — The system extracts all readable text from the PDF pages
3. **Text Chunking** — Long documents are split into overlapping chunks for better context
4. **Embedding & Retrieval** — Text chunks are embedded and the most relevant chunks are retrieved for each question
5. **LLM Answer Generation** — Retrieved context is passed to an LLM which generates a precise answer

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| PyPDF2 / pdfplumber | PDF text extraction |
| LangChain | LLM chaining and retrieval pipeline |
| OpenAI / Gemini API | Language model for answer generation |
| Streamlit | Web-based user interface |
| FAISS | Vector store for document retrieval |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/arunkumararavindhakshan05-sudo/Pdf-Reader-Chatbot.git
cd Pdf-Reader-Chatbot

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, upload a PDF, and start asking questions!

---

## 📁 Project Structure

```
Pdf-Reader-Chatbot/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Support multiple PDF uploads at once
- [ ] Add chat history / memory
- [ ] Support Word (.docx) and Excel (.xlsx) files
- [ ] Deploy on Hugging Face Spaces or Streamlit Cloud

---

## 👤 Author

**Arunkumar Aravindhakshan**
🔗 [LinkedIn](https://linkedin.com/in/arunkumar-aravindhakshan) | [GitHub](https://github.com/arunkumararavindhakshan05-sudo)
