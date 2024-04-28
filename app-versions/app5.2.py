# sidebar problem solved

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from nltk import ngrams
from nltk.tokenize import word_tokenize

load_dotenv()
os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Commented this line as it seems unnecessary

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Use instructor embeddings for vectorization
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # print(embeddings)
    # if not embeddings:
    #     print("Error: Embeddings is empty or has no elements.")
    #     return

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Context:\n {context}?\n
def get_conversational_chain(use_gemini_knowledge):
    
    if use_gemini_knowledge:
        prompt_template = """
        You can answer the question solely on the basis of pretrained information. Don't use the context given\n\n
        Context = {"context": "0"}  # Embed the context with zeros
        Question: \n{question}\n

        Answer:
        """
        # context = {"context": "0"}  # Embed the context with zeros
    else:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        # context = {"context": "{context}"}
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def compute_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def compute_accuracy_jaccard(generated_answer, ground_truth_answer):
    tokens_generated = set(word_tokenize(generated_answer.lower()))
    tokens_ground_truth = set(word_tokenize(ground_truth_answer.lower()))
    jaccard_similarity = compute_jaccard_similarity(tokens_generated, tokens_ground_truth)
    return jaccard_similarity

def user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge):
    # Use instructor embeddings for user input as well
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if not os.path.exists("faiss_index/index.faiss"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain(use_gemini_knowledge)

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    generated_answer = response["output_text"]

    accuracy_jaccard = compute_accuracy_jaccard(generated_answer, ground_truth_answer)

    print(response)
    st.write("Generated Answer: ", generated_answer)

    st.success(f"Jaccard Similarity Accuracy: {accuracy_jaccard:.2%}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Kya hukum hai mere aaka ~Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        ground_truth_answer = st.text_input("Enter Ground Truth Answer for Evaluation")
        use_gemini_knowledge = st.checkbox("Use Gemini's Knowledge")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                # user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge)
    
    if user_question and pdf_docs and ground_truth_answer:
        user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge)
    else:
        st.warning("Please provide input for all fields")

if __name__ == "__main__":
    main()