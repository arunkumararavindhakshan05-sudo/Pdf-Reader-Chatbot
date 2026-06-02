import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.write("Upload ONE PDF file and ask it any question!")
st.info("Upload only one PDF at a time. PDF must contain text, not scanned images.")

pdf = st.file_uploader("Upload your PDF here", type="pdf", accept_multiple_files=False)

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
                except Exception:
                    continue

        if not text.strip():
            st.error("Could not read text from this PDF. Please try a different PDF that contains actual text.")
            st.stop()

        word_count = len(text.split())
        st.success("PDF loaded! Found " + str(word_count) + " words. Ask me anything!")

        question = st.text_input("Type your question here and press Enter:")

        if question:
            with st.spinner("Finding your answer..."):
                splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                context = "\n\n".join(chunks[:3])

                try:
                    llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        temperature=0,
                        api_key=os.environ.get("GROQ_API_KEY")
                    )
                    prompt = ChatPromptTemplate.from_template(
                        "You are a helpful assistant. Use the document below to answer the question.\n"
                        "If the answer is not in the document, say I could not find that in the PDF.\n\n"
                        "Document:\n{context}\n\n"
                        "Question: {question}\n\n"
                        "Answer:"
                    )
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({"context": context, "question": question})
                    st.write("### Answer:")
                    st.write(answer)
                    st.write("---")
                    st.caption("Ask another question in the box above!")

                except Exception as e:
                    st.error("AI error. Please try again in a few seconds.")

    except Exception as e:
        st.error("Could not read this PDF. Please try a different file.")
