# PDF Reader Chatbot

A Streamlit application for asking questions about PDF, Word, Excel, CSV, PowerPoint and text documents using text or voice, with personal-data masking and prompt-injection detection.

The application extracts text from an uploaded document, divides it into smaller sections, and retrieves the sections most relevant to the user’s question. A Groq-hosted language model then generates an answer using the retrieved document content.

[Open the live application](https://arun-pdf-chatbot.streamlit.app/)

## Features

* Upload and process PDF, Word (`.docx`), Excel (`.xlsx`), CSV, PowerPoint (`.pptx`), text and Markdown files
* Cite answers by source location, such as a page, a slide, or a sheet row range
* Mask personal data before any text reaches the language model, then restore real values in the answer:
  * names and places with Microsoft Presidio (spaCy `en_core_web_sm`)
  * Indian postal addresses with a PIN code, and every value in spreadsheet columns such as Name, Customer or Address
  * email, phone, Aadhaar, PAN, IFSC, card numbers and IP addresses with checksum-validated patterns
* Answer spreadsheet calculations (totals, averages, counts, top-N, group-by) exactly with pandas across all rows
* Detect prompt-injection attempts hidden in documents, including zero-width characters, and warn the user
* Ask questions using text input
* Record questions using a microphone
* Convert recorded questions to text with Groq Whisper
* Retrieve relevant document sections using semantic search
* Generate answers based on the uploaded document
* Ask questions by voice in 16 languages, including Tamil, Hindi, Malayalam, Telugu and Kannada, and get answers in the same language
* Correct the transcription before the question is sent
* Listen to generated answers using text-to-speech: Groq voices for English, or Azure AI Speech for multilingual answers
* Choose from multiple available voices
* Run automated tests for document processing, retrieval, RAG, and voice features

## How It Works

1. The user uploads a document through the Streamlit interface.
2. A loader for the file type extracts its text (`pypdf`, `python-docx`, `openpyxl`, `python-pptx` or the CSV module). Spreadsheet rows are grouped into blocks that repeat the header row.
3. The text is scanned for prompt-injection patterns, and invisible characters are removed.
4. The extracted text is divided into overlapping chunks to preserve context.
5. Sentence-transformer embeddings are created for each chunk.
6. The embeddings are stored in a FAISS index.
7. When the user asks a question, the most relevant chunks are retrieved.
8. With privacy masking on, personal data in the question and retrieved content is replaced with placeholders. The masked content is sent to the Groq-hosted language model.
9. Placeholders in the answer are restored locally and the answer is displayed.

For Excel and CSV files, calculation questions such as "total amount for South" take a different path. The model returns a JSON query plan (filters, group-by, aggregations, sorting) that is validated against the real columns, pandas runs it over every row, and the model explains the exact result. The model never writes or runs code. in the application.
9. If required, the answer can also be converted to speech and played back.

## Technology Used

| Technology            | Usage                                                     |
| --------------------- | --------------------------------------------------------- |
| Python                | Application development                                   |
| Streamlit             | User interface                                            |
| pypdf                 | PDF text extraction                                       |
| Sentence Transformers | Text embeddings                                           |
| FAISS                 | Semantic document retrieval                               |
| LangChain             | Text splitting and Groq model integration                 |
| Groq API              | Answer generation, speech recognition, and text-to-speech |
| pytest                | Automated testing                                         |
| Ruff                  | Code formatting and linting                               |
| uv                    | Dependency and environment management                     |

## Project Structure

```text
Pdf-Reader-Chatbot/
├── src/
│   ├── azure_speech.py
│   ├── document_processor.py
│   ├── injection_guard.py
│   ├── loaders.py
│   ├── ner.py
│   ├── pii_redactor.py
│   ├── rag_service.py
│   ├── retriever.py
│   ├── table_engine.py
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
* `loaders.py` picks a loader by file extension and returns location-labelled chunks for every supported format.
* `pii_redactor.py` masks and restores personal data using patterns plus Luhn and Verhoeff checksum validation.
* `ner.py` loads Microsoft Presidio for name and place detection, and falls back to pattern masking when it is not installed.
* `table_engine.py` plans, validates and runs spreadsheet calculations with pandas.
* `injection_guard.py` flags instruction-like text and strips invisible characters from documents.
* `retriever.py` creates embeddings and retrieves relevant document sections.
* `rag_service.py` sends the retrieved context to the language model and generates answers.
* `voice_service.py` handles speech-to-text and English text-to-speech with Groq.
* `azure_speech.py` reads answers aloud in other languages with Azure AI Speech, choosing a voice that matches the answer's language.
* `summarizer.py` summarises a document once at upload and stores the summary in the index, so questions about the whole document can be answered.
* `tests/` contains automated tests for the main application components.

## Local Setup

### Prerequisites

Before running the project, make sure the following are available:

* Python 3.12 or newer
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

Copy `.env.example` to `.env`, then add the Groq API key:

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

* The application processes one document at a time.
* Scanned PDFs and images without readable text may require OCR support.
* Name detection uses spaCy's small English model, which misses some names (especially Indian names in free text). Spreadsheet name and address columns are always masked.
* Addresses in free text are masked only when they include an Indian PIN code.
* The spreadsheet engine treats the first non-empty row of each sheet as the header.
* Answer quality depends on the text available in the uploaded document.
* A valid Groq API key is required for language and voice features.
* Multilingual spoken answers need `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`; without them, spoken answers are English-only.

## Planned Improvements

* Support multiple PDF files in one session
* Add OCR support for scanned documents
* Evaluate masking recall on a labelled test set and try a larger spaCy or transformer model
* Edit Word and Excel files by voice command with preview and confirmation
* Preserve conversation history during a session
* Allow users to download their chat history

## Author

**Arunkumar Aravindhakshan**

* [GitHub profile](https://github.com/arunkumararavindhakshan05-sudo)
* [LinkedIn profile](https://linkedin.com/in/arunkumar-aravindhakshan)
