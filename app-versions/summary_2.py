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
from langchain_core.language_models import base 
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from langchain_core import *
from langchain.memory.summary import ConversationSummaryMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")

priming_variable = "You are a friendly expert that provides in-detail answers with reference to the questions in the 'passed' domain."
style_and_tone_variable = "Use 10th-grade language and explain things in a simple way that anyone can understand"
    

def preprocess_question(user_question, domain, tone):
    # Identify variables based on domain
    priming_variable = "You are an expert that provides in-detail answers with reference to the questions in the {} domain.".format(domain)
    if tone is not None:
        style_and_tone_variable = tone 
    else:
        style_and_tone_variable = "Use 10th-grade language and explain things in a simple way that anyone can understand"
        
    processed_question = {
        "response": user_question,
        "priming": priming_variable,
        "style_and_tone_instructions": style_and_tone_variable
    }

    # Concatenate all strings into a single string
    concatenated_string = user_question + " " + priming_variable + " " + style_and_tone_variable + "."

    # Adding the concatenated string to the dictionary
    processed_question['concatenated_string'] = concatenated_string

    #print(processed_question)
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
    return vector_store  # Add this line to return the vector_store

def get_vector_store_for_filtered_docs(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("filtered_faiss_index")
    return vector_store  # Add this line to return the vector_store
    
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



def user_input(user_question, pdf_docs, ground_truth_answer, domain):
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    memory=ConversationSummaryMemory(llm=llm, return_messages=True)
    summary_memory_chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
        )
    
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
 
    new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    prompt_text = """from context below return text chunks relevant to the question : {question}? Context: {context} """ 
    relevance_score = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3),chain_type="stuff",
                                  prompt= PromptTemplate(template=prompt_text, input_variables=["question", "context"]))
    result = relevance_score({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    filtered_docs = result["output_text"]
 
    st.write("Filtered Doc: ", filtered_docs)
    
    filtered_text_chunks = get_text_chunks(filtered_docs)
    get_vector_store_for_filtered_docs(filtered_text_chunks)
    new_filtered_docs = FAISS.load_local("filtered_faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    filtered_documents = new_filtered_docs.similarity_search(user_question)     
    
    summary = memory
    st.write ("memory: ",memory)
    prompt_template = """     Context:
                              {context}?
    
                              conversation summary:\n 
                              {summary}
                              
                              Question: 
                              {question}
                              
                              Answer:    
                      """
                      
    chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7), chain_type="stuff",
                          prompt=PromptTemplate(template=prompt_template, input_variables=["question","context","summary"]))


    # Pass documents as input to LangChain
    response = chain({"question": user_question, "input_documents": filtered_documents, "summary": summary}, return_only_outputs=True)
    generated_answer = response["output_text"]
    memory.save_context({"input": user_question}, {"output": generated_answer})
    
    
    history_print = memory.load_memory_variables({})
    # fil_content= filtered_documents['page_content']
    prev = f"Context: {filtered_documents} \n conversation summary: {history_print} \n question: {user_question}"
    summary_memory_chain.predict(input = prev)
    
     
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

    st.success("Evaluation Metrics:")
    for point in bullet_points:
        st.success(f"- {point}")

    # st.success("Evaluation Metrics (JSON Format):")
    # st.success(json_format)
    
    return output
    
    
def main():
    st.set_page_config("PhatakaReader Pro")
    st.header("Ask me anything ~Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    domain = st.text_input("Enter the domain (e.g., plumbing, history)")
    tone =st.text_input("Set the tone for your responses. For example, explain things in a simple way that anyone can understand (default).")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        ground_truth_answer = st.text_input("Enter Ground Truth Answer for Evaluation")
        # use_gemini_knowledge = st.checkbox("Use Gemini only :)")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if user_question and pdf_docs and ground_truth_answer and domain:
                    processed_question = preprocess_question(user_question, domain,tone)
                    st.write("Preprocessed question: ",processed_question)
                    output = user_input(processed_question['concatenated_string'], pdf_docs, ground_truth_answer, domain)
                    st.write("Response:", output["response"])                

                else:
                    st.warning("Please provide input for all fields")
                    
    # st.write("Response:", output["response"])                

if __name__ == "__main__":
    main() 