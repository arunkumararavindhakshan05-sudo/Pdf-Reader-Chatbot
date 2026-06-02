import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

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

    # Cut into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # Free offline embeddings
    st.info("Loading AI model... please wait on first run.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    st.success("PDF loaded successfully! Ask me anything about it.")

    question = st.text_input("Type your question here:")

    if question:
        docs = knowledge_base.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Groq - completely free and very fast!
        llm = ChatGroq(
              model="llama-3.1-8b-instant",
              temperature=0
   )
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the context below to answer the question.
        If the answer is not in the context, say "I could not find that in the PDF."

        Context:
        {context}

        Question:
        {question}
        """)

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        st.write("### Answer:")
        st.write(answer)