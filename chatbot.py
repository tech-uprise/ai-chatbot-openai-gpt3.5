import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# add your OPENAI API KEY
OPENAI_API_KEY = "<< add your OPENAI API KEY " #new key


# upload PDF files
st.header("My First chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

    text = ""
    if file is not None:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        # Uncomment the following line if you want to see the extracted text
        st.write(text)

# Break it into chunks outside the sidebar
if text:
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # Uncomment the following line if you want to see the chunks
    st.write(chunks)

    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS (Facebook AI Symantic Search)
    # Assuming from_texts is a valid method and embeddings is correctly instantiated
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user question
    user_question = st.text_input("Type your question here")

    # do similarity search
    if user_question:
        # Assuming similarity_search is a valid method and returns a result that can be displayed
        match = vector_store.similarity_search(user_question)
        st.write(match)
        #define the llm
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0, # asking the LLM, the response to be not random, the lower the value, asking for lower random value
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        # output results
        #chain -> take the question, get the relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
