import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import faiss
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# genai.configure("appppppi key")


def get_pdf_text(pdf_bytes):
    text = ""
    
    # Create a BytesIO object from the bytes-like object
    pdf_stream = io.BytesIO(pdf_bytes)
    
    # Use the BytesIO object as the stream for PdfReader
    pdf_reader = PdfReader(pdf_stream)
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter(text)
    return chunks

def get_vector_store(text_chunks):
    print("Indexing process started...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Indexing process completed.")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context. if
    provided context not useful, dont provide wrong answer. just say that the answer is not in the context \n\n
    Context: \n {context}?\n
    Question: \n{question}?\n
    
    Answer: 
    """
    
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question,pdf_docs):
    print("Index loading started...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    index_path = "faiss_index"
    if not os.path.exists(index_path):
        # If the index file doesn't exist, create it
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    # Load the Faiss index
    new_db = FAISS.load_local(index_path, embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])
    print("Index loading completed.")
        

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Gemini ")
    user_question = st.text_input("Ask a question from the PDF files")

    # Move the file uploader inside the main if block
    if user_question:
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit button")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs.read())
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
            
            # Pass pdf_docs to user_input function
            user_input(user_question, pdf_docs)

    with st.sidebar:
        st.title("Menu:")
        # Remove the pdf_docs uploader from the sidebar

if __name__ == "__main__":
    main()
