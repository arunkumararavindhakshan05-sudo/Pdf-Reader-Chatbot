import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.write("Upload any PDF and ask it questions!")

pdf = st.file_uploader("Upload your PDF here", type="pdf")

if pdf is not None:

    # Read the PDF
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        st.error("Could not read text from this PDF. Try a different PDF.")
        st.stop()

    # Cut into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Lightweight embeddings - works on any server!
    embeddings = FakeEmbeddings(size=768)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    st.success("PDF loaded! Ask me anything about it.")

    question = st.text_input("Type your question here:")

    if question:
        docs = knowledge_base.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use ONLY the context below to answer.
        If the answer is not in the context, say "I could not find that in the PDF."

        Context:
        {context}

        Question:
        {question}

        Answer clearly and concisely:
        """)

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Thinking..."):
            answer = chain.invoke({
                "context": context,
                "question": question
            })

        st.write("### Answer:")
        st.write(answer)
