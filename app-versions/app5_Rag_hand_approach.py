# File "C:\Users\VIBGYOR\Desktop\LLM-Mini-Resources\app5_Rag_copy.py", line 115, in filter_text_chunks
#     user_question_vector = vector_store.encode([user_question])[0]
#                            ^^^^^^^^^^^^^^^^^^^
# modfiy this code for the same:

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
from nltk.tokenize import word_tokenize
from rouge import Rouge
from sklearn.metrics import f1_score
import time
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
# Remove unnecessary import: from nltk.translate import meteor
# No need for imports you're not using: from langchain_core import StopCandidateException
from langchain_core import *
# from langchain_core import StopCandidateException
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")

def preprocess_question(user_question, domain):
    # Identify variables based on domain
    priming_variable = "You are an expert that provide in-detail answers with reference to the questions in the {} domain.".format(domain)
    style_and_tone_variable = "Use 10th-grade language and explain things in a simple way that anyone can understand."
    processed_question = {
        "response": user_question,
        "priming": priming_variable,
        "style_and_tone_instructions": style_and_tone_variable
    }
    return processed_question

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

def compute_meteor_score(reference, hypothesis):
    reference_tokens = word_tokenize(reference)  # Tokenize the reference here
    hypothesis_tokens = word_tokenize(hypothesis)
    score = meteor_score([reference_tokens], hypothesis_tokens)
    return score

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


def generate_output(generated_answer, metrics):
    output = {"response": generated_answer}
    output.update(metrics)

    bullet_points = [
        f"Jaccard Similarity Accuracy: {metrics['accuracy_jaccard']:.2%}",
        f"BLEU Score: {metrics['bleu_score']:.4f}",
        f"METEOR Score: {metrics['meteor_score']:.4f}",
        f"ROUGE Scores: {metrics['rouge_scores']}",
        f"F1 Score: {metrics['f1_score']:.4f}",
    ]

    json_format = json.dumps(output, indent=4)

    return output, bullet_points, json_format


def filter_text_chunks(docs, user_question):
    # Extract text content based on object structure
    docs_text_content = docs
    # for doc in docs:
    #     if isinstance(doc, str):  # Plain text string
    #         docs_text_content.append(doc.strip())
    #     else:  # Assume a Document object with a 'content' property (modify if different)
    #         try:
    #             text_content = doc.content.strip()  # Assuming 'content' property holds text
    #         except AttributeError:
    #             # Handle cases where 'content' property might not exist
    #             text_content = ""
    #         docs_text_content.append(text_content)

    # Leverage Gemini to directly filter relevant documents
    # Chain setup
    prompt_template = """Filter these documents to keep only those that are relevant to the question: {question}\n\n{input_documents}"""
    
    chain = load_qa_chain(
        ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3),
        chain_type = "stuff",
        prompt=PromptTemplate(template=prompt_template, input_variables = ["question", "input_documents"]))

    # Call the chain with the prompt and user question
    response = chain({"question": user_question,"input_documents": docs_text_content}, return_only_outputs=True)
    filtered_docs = response["output_text"]
    filtered_docs = filtered_docs.strip().split("\n")
    return filtered_docs



def user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge, domain):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)

    # Perform similarity search to retrieve relevant text chunks
    new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Add an optimization step to filter the most appropriate retrieved text chunks
    filtered_docs = filter_text_chunks(docs, user_question)

    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3), chain_type="stuff",
                          prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]))
    
    # prompt_template = """
    # Context:\n {context}?\n
    # Question: \n{user_question}\n

    # Answer:
    # """

    # chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3), chain_type="stuff",
    #                       prompt=PromptTemplate(input_variables=["context", "user_question"], template=prompt_template))


    # response = chain({"context": filtered_docs, "question": user_question}, return_only_outputs=True)    
    response = chain({"context":docs, "question": user_question}, return_only_outputs=True)
    generated_answer = response["output_text"]
        
    # except StopCandidateException as e:
    #     generated_answer = f"Generation stopped: {e.finish_reason}"

    accuracy_jaccard = compute_accuracy_jaccard(generated_answer, ground_truth_answer)
    bleu_score = compute_bleu_score(ground_truth_answer, generated_answer)
    meteor_score = compute_meteor_score(ground_truth_answer, generated_answer)
    rouge_scores = compute_rouge_scores(ground_truth_answer, generated_answer)
    f1_score_value = compute_f1_score(ground_truth_answer, generated_answer)

    output, bullet_points, json_format = generate_output(generated_answer, {
        "accuracy_jaccard": accuracy_jaccard,
        "bleu_score": bleu_score,
        "meteor_score": meteor_score,
        "rouge_scores": rouge_scores,
        "f1_score": f1_score_value,
    })

    st.write("Processed Question:")
    st.write("Response:", output["response"])
    st.write("Priming:", output.get("priming", ""))
    st.write("Style and Tone Instructions:", output.get("style_and_tone_instructions", ""))

    st.success("Evaluation Metrics:")
    for point in bullet_points:
        st.success(f"- {point}")

    st.success("Evaluation Metrics (JSON Format):")
    st.success(json_format)

def main():
    st.set_page_config("Chat PDF")
    st.header("Ask me anything ~Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    domain = st.text_input("Enter the domain (e.g., plumbing, history)")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        ground_truth_answer = st.text_input("Enter Ground Truth Answer for Evaluation")
        use_gemini_knowledge = st.checkbox("Use Gemini's Knowledge")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if user_question and pdf_docs and ground_truth_answer and domain:
                    processed_question = preprocess_question(user_question, domain)
                    user_input(processed_question["response"], pdf_docs, ground_truth_answer, use_gemini_knowledge, domain)
                else:
                    st.warning("Please provide input for all fields")

if __name__ == "__main__":
    main()