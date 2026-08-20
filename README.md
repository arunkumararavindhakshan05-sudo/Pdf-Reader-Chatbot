# PDF Reader Chatbot

A Streamlit application for asking questions about PDF documents using text or voice.

The application extracts text from an uploaded PDF, divides it into smaller sections, and retrieves the sections most relevant to the user’s question. A Groq-hosted language model then generates an answer using the retrieved document content.

[Open the live application](https://arun-pdf-chatbot.streamlit.app/)

## Features

* Upload and process PDF documents
* Ask questions using text input
* Record questions using a microphone
* Convert recorded questions to text with Groq Whisper
* Retrieve relevant document sections using semantic search
* Generate answers based on the uploaded document
* Listen to generated answers using text-to-speech
* Choose from multiple available voices
* Validate document processing, retrieval, RAG, and voice services through automated tests

## How It Works

1. The user uploads a PDF through the Streamlit interface.
2. Text is extracted from the document using `pypdf`.
3. The extracted text is divided into overlapping chunks to preserve context.
4. Sentence-transformer embeddings are created for each chunk.
5. The embeddings are stored in a FAISS index.
6. When the user asks a question, the most relevant chunks are retrieved.
7. The retrieved content and question are sent to the Groq-hosted language model.
8. The generated answer is displayed in the application.
9. If required, the answer can also be converted to speech and played back.

## Technology Used

| Technology            | Usage                                                     |
| --------------------- | --------------------------------------------------------- |
| Python                | Application development                                   |
| Streamlit             | User interface                                            |
| pypdf                 | PDF text extraction                                       |
| Sentence Transformers | Text embeddings                                           |
| FAISS                 | Semantic document retrieval                               |
| LangChain             | RAG workflow                                              |
| Groq API              | Answer generation, speech recognition, and text-to-speech |
| pytest                | Automated testing                                         |
| Ruff                  | Code formatting and linting                               |
| uv                    | Dependency and environment management                     |

## Project Structure

```text
Pdf-Reader-Chatbot/
├── src/
│   ├── document_processor.py
│   ├── rag_service.py
│   ├── retriever.py
│   └── voice_service.py
├── tests/
├── .gitignore
├── app.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

### Main Components

* `app.py` contains the Streamlit user interface and connects the application services.
* `document_processor.py` extracts and divides PDF text into chunks.
* `retriever.py` creates embeddings and retrieves relevant document sections.
* `rag_service.py` sends the retrieved context to the language model and generates answers.
* `voice_service.py` handles speech-to-text and text-to-speech operations.
* `tests/` contains automated tests for the main application components.

## Local Setup

### Prerequisites

Before running the project, make sure the following are available:

* Python 3
* Git
* A Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/arunkumararavindhakshan05-sudo/Pdf-Reader-Chatbot.git
cd Pdf-Reader-Chatbot
```

### 2. Install the Dependencies

Using `uv`:

```bash
uv sync
```

Alternatively, install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Configure the API Key

Create a `.env` file in the project directory and add the Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit the `.env` file or expose the API key in the source code.

### 4. Run the Application

Using `uv`:

```bash
uv run streamlit run app.py
```

Or run it directly:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## Running the Tests

Using `uv`:

```bash
uv run pytest
```

Or:

```bash
pytest
```

## Checking Code Quality

Using `uv`:

```bash
uv run ruff check .
```

Or:

```bash
ruff check .
```

## Current Limitations

* The application processes one PDF at a time.
* Scanned PDFs without readable text may require OCR support.
* Answer quality depends on the text available in the uploaded document.
* A valid Groq API key is required for language and voice features.

## Planned Improvements

* Support multiple PDF files in one session
* Add OCR support for scanned documents
* Preserve conversation history during a session
* Support additional document formats
* Add source references to generated answers

## Author

**Arunkumar Aravindhakshan**

* [GitHub profile](https://github.com/arunkumararavindhakshan05-sudo)
* [LinkedIn profile](https://linkedin.com/in/arunkumar-aravindhakshan)
