import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import nltk
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from nltk import ngrams, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from sklearn.metrics import f1_score
import time
# from langchain_core import StopCandidateException
from langchain_core import *
from nltk.translate import meteor
# nltk.download('wordnet')

load_dotenv()
os.getenv("GOOGLE_API_KEY")

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(prompt_template):
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

def compute_bleu_score(reference, hypothesis):
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    return sentence_bleu([reference_tokens], hypothesis_tokens)

# def compute_meteor_score(reference, hypothesis):
#     def preprocess(text):
#         return word_tokenize(text.lower())

#     reference_tokens = preprocess(reference)
#     hypothesis_tokens = preprocess(hypothesis)

#     return single_meteor_score([reference_tokens], hypothesis_tokens)


# def compute_meteor_score(reference_str, hypothesis_str):
#     meteor_obj = meteor.Meteor()
#     score = meteor_obj.single_meteor_score(reference_str, hypothesis_str)
#     return score

#     # Calculate the METEOR score
#     score = single_meteor_score(reference_str, hypothesis_str)
#     return score

# def single_meteor_score(reference, hypothesis):
#     meteor_obj = meteor.Meteor()
#     score = meteor_obj.single_meteor_score(reference, hypothesis)
#     return score

def compute_rouge_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores  # Assuming a single reference

def compute_f1_score(reference, hypothesis):
    reference_tokens = set(word_tokenize(reference.lower()))
    hypothesis_tokens = set(word_tokenize(hypothesis.lower()))
    intersection = len(reference_tokens.intersection(hypothesis_tokens))
    precision = intersection / len(hypothesis_tokens) if len(hypothesis_tokens) > 0 else 0
    recall = intersection / len(reference_tokens) if len(reference_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def generate_response_with_retry(model, input_data, max_retries=3):
    for _ in range(max_retries):
        try:
            response = model.generate_response(input_data)
            return response
        except StopCandidateException as e:
            # Handle the stop condition, optionally log the exception
            time.sleep(1)  # Add a delay before retrying
            continue
    # If max retries exceeded, raise an exception or return a default response
    raise Exception("Failed to generate response after maximum retries")

    # Usage
    try:
        # Code that may raise StopCandidateException
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        generated_answer = response["output_text"]
    except StopCandidateException as e:
        # Handle the StopCandidateException, e.g., log it or provide a default response
        generated_answer = "Generation stopped: {}".format(e.finish_reason)
        # You can also choose to re-raise the exception if you want to propagate it further.
        # raise


def user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if not os.path.exists("faiss_index/index.faiss"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    if use_gemini_knowledge:
        prompt_template = """
        You can answer the question based on the provided context, or you can use your own knowledge if necessary.\n\n 
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    else:
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

    chain = get_conversational_chain(prompt_template)

    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        generated_answer = response["output_text"]
    except StopCandidateException as e:
        # Handle the StopCandidateException, e.g., log it or provide a default response
        generated_answer = f"Generation stopped: {e.finish_reason}"
  
    # Evaluation Metrics
    accuracy_jaccard = compute_accuracy_jaccard(generated_answer, ground_truth_answer)
    bleu_score = compute_bleu_score(ground_truth_answer, generated_answer)
    # meteor_score = compute_meteor_score(ground_truth_answer, generated_answer)
    rouge_scores = compute_rouge_scores(ground_truth_answer, generated_answer)
    f1_score_value = compute_f1_score(ground_truth_answer, generated_answer)

    # Print Metrics
    st.write("Generated Answer: ", generated_answer)
    st.success(f"Jaccard Similarity Accuracy: {accuracy_jaccard:.2%}")
    st.success(f"BLEU Score: {bleu_score:.4f}")
    # st.success(f"METEOR Score: {meteor_score:.4f}")
    st.success(f"ROUGE Scores: {rouge_scores}")
    st.success(f"F1 Score: {f1_score_value:.4f}")

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
                if user_question and pdf_docs and ground_truth_answer:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge)
                else:
                    st.warning("Please provide input for all fields")

if __name__ == "__main__":
    main()
