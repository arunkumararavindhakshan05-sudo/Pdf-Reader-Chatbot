import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.write("Upload ONE PDF file and ask it any question!")
st.info("Tip: Upload only one PDF at a time. File must contain text (not scanned images).")

pdf = st.file_uploader(
    "Upload your PDF here",
    type="pdf",
    accept_multiple_files=False
)

if pdf is not None:
    try:
        with st.spinner("Reading your PDF... please wait"):
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                except:
                    continue

        if not text.strip():
            st.error("""
Could not read text from this PDF.
This usually happens when the PDF is a scanned image.
Please try a different PDF that contains actual text.
            """)
            st.stop()

        word_count = len(text.split())
        st.success(f"PDF loaded successfully! Found {word_count} words. Now ask your question below!")

        question = st.text_input("Type your question here and press Enter:")

        if question:
            with st.spinner("Finding your answer..."):

                splitter = CharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200
                )
                chunks = splitter.split_text(text)
                context = "\n\n".join(chunks[:3])

                try:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        api_key=os.environ.get("GROQ_API_KEY")
                    )

                    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Read the document content below and answer the question.
If you cannot find the answer in the document, say "I could not find that in the PDF."

Document content:
{context}

Question: {question}

Give a clear and helpful answer:
""")
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })

                    st.write("### Answer:")
                    st.write(answer)

                    st.write("---")
                    st.write("Ask another question above!")

                except Exception as e:
                    st.error(f"AI error: {str(e)}")
                    st.write("Please try again in a few seconds.")

    except Exception as e:
        st.error(f"Could not read this PDF: {str(e)}")
        st.write("Please try uploading a different PDF file.")import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.write("Upload ONE PDF file and ask it any question!")
st.info("Tip: Upload only one PDF at a time. File must contain text (not scanned images).")

pdf = st.file_uploader(
    "Upload your PDF here",
    type="pdf",
    accept_multiple_files=False
)

if pdf is not None:
    try:
        with st.spinner("Reading your PDF... please wait"):
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                except:
                    continue

        if not text.strip():
            st.error("""
Could not read text from this PDF.
This usually happens when the PDF is a scanned image.
Please try a different PDF that contains actual text.
            """)
            st.stop()

        word_count = len(text.split())
        st.success(f"PDF loaded successfully! Found {word_count} words. Now ask your question below!")

        question = st.text_input("Type your question here and press Enter:")

        if question:
            with st.spinner("Finding your answer..."):

                splitter = CharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200
                )
                chunks = splitter.split_text(text)
                context = "\n\n".join(chunks[:3])

                try:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        api_key=os.environ.get("GROQ_API_KEY")
                    )

                    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Read the document content below and answer the question.
If you cannot find the answer in the document, say "I could not find that in the PDF."

Document content:
{context}

Question: {question}

Give a clear and helpful answer:
""")
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })

                    st.write("### Answer:")
                    st.write(answer)

                    st.write("---")
                    st.write("Ask another question above!")

                except Exception as e:
                    st.error(f"AI error: {str(e)}")
                    st.write("Please try again in a few seconds.")

    except Exception as e:
        st.error(f"Could not read this PDF: {str(e)}")
        st.write("Please try uploading a different PDF file.")import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.write("Upload ONE PDF file and ask it any question!")
st.info("Tip: Upload only one PDF at a time. File must contain text (not scanned images).")

pdf = st.file_uploader(
    "Upload your PDF here",
    type="pdf",
    accept_multiple_files=False
)

if pdf is not None:
    try:
        with st.spinner("Reading your PDF... please wait"):
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                except:
                    continue

        if not text.strip():
            st.error("""
Could not read text from this PDF.
This usually happens when the PDF is a scanned image.
Please try a different PDF that contains actual text.
            """)
            st.stop()

        word_count = len(text.split())
        st.success(f"PDF loaded successfully! Found {word_count} words. Now ask your question below!")

        question = st.text_input("Type your question here and press Enter:")

        if question:
            with st.spinner("Finding your answer..."):

                splitter = CharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200
                )
                chunks = splitter.split_text(text)
                context = "\n\n".join(chunks[:3])

                try:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        api_key=os.environ.get("GROQ_API_KEY")
                    )

                    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Read the document content below and answer the question.
If you cannot find the answer in the document, say "I could not find that in the PDF."

Document content:
{context}

Question: {question}

Give a clear and helpful answer:
""")
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })

                    st.write("### Answer:")
                    st.write(answer)

                    st.write("---")
                    st.write("Ask another question above!")

                except Exception as e:
                    st.error(f"AI error: {str(e)}")
                    st.write("Please try again in a few seconds.")

    except Exception as e:
        st.error(f"Could not read this PDF: {str(e)}")
        st.write("Please try uploading a different PDF file.")import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.write("Upload ONE PDF file and ask it any question!")
st.info("Tip: Upload only one PDF at a time. File must contain text (not scanned images).")

pdf = st.file_uploader(
    "Upload your PDF here",
    type="pdf",
    accept_multiple_files=False
)

if pdf is not None:
    try:
        with st.spinner("Reading your PDF... please wait"):
            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                try:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
                except:
                    continue

        if not text.strip():
            st.error("""
Could not read text from this PDF.
This usually happens when the PDF is a scanned image.
Please try a different PDF that contains actual text.
            """)
            st.stop()

        word_count = len(text.split())
        st.success(f"PDF loaded successfully! Found {word_count} words. Now ask your question below!")

        question = st.text_input("Type your question here and press Enter:")

        if question:
            with st.spinner("Finding your answer..."):

                splitter = CharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=200
                )
                chunks = splitter.split_text(text)
                context = "\n\n".join(chunks[:3])

                try:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        api_key=os.environ.get("GROQ_API_KEY")
                    )

                    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Read the document content below and answer the question.
If you cannot find the answer in the document, say "I could not find that in the PDF."

Document content:
{context}

Question: {question}

Give a clear and helpful answer:
""")
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context,
                        "question": question
                    })

                    st.write("### Answer:")
                    st.write(answer)

                    st.write("---")
                    st.write("Ask another question above!")

                except Exception as e:
                    st.error(f"AI error: {str(e)}")
                    st.write("Please try again in a few seconds.")

    except Exception as e:
        st.error(f"Could not read this PDF: {str(e)}")
        st.write("Please try uploading a different PDF file.")
