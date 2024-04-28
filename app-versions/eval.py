import csv
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
from langchain_core import *
from langchain.memory.summary import ConversationSummaryMemory
from langchain.chains import ConversationChain
import requests
import pdfkit
from vertexai import generative_models
import pandas as pd
from bs4 import BeautifulSoup
from google.generativeai.types import safety_types
import google.generativeai as genai


load_dotenv()
os.getenv("GOOGLE_API_KEY")

priming_variable = "You are a friendly expert that provides in-detail answers with reference to the questions in the 'passed' domain."
style_and_tone_variable = "Use 10th-grade language and explain things in a simple way that anyone can understand"

def handle_safety_flag(csv_index):
    print("Safety flag triggered. Moving to the next question.")
    next_question_index = csv_index + 1  # Assuming csv_index is the current question's index
    with open("C:\\Users\\VIBGYOR\\Desktop\\LLM-Mini-Resources\\TruthfulQA-main-BenchmarkNo.1\\TruthfulQA-main\\TruthfulQATesting_Phataka.csv") as csvfile:
        reader = csv.reader(csvfile)
        next_question_row = next(reader, None)  # Skip to the next row
        if next_question_row:
            next_question_text = next_question_row[0]  # Assuming question text is in the first column
            # Process the next question
            preprocess_question(next_question_text)
        else:
            print("No more questions in the CSV.")

def preprocess_question(user_question, domain, tone):
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

    concatenated_string = user_question + " " + priming_variable + " " + style_and_tone_variable + "."
    processed_question['concatenated_string'] = concatenated_string
    return processed_question

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store 

def get_vector_store_for_filtered_docs(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("filtered_faiss_index")
    return vector_store 
    
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
#     reference_tokens = word_tokenize(reference) 
#     hypothesis_tokens = word_tokenize(hypothesis)
#     score = meteor_score([reference_tokens], hypothesis_tokens)
#     return score

def compute_rouge_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores 

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
        # f"METEOR Score: {metrics['meteor_score']:.4f}",
        f"ROUGE Scores: {metrics['rouge_scores']}",
        f"F1 Score: {metrics['f1_score']:.4f}",
    ]

    json_format = json.dumps(output, indent=4)

    return output, bullet_points, json_format

def user_input(user_question, raw_text, ground_truth_answer, domain, csv_index):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1,
                    safety_settings= {generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH})
        
        memory = ConversationSummaryMemory(llm=llm, return_messages=True)
        summary_memory_chain = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )

        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                                  allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        prompt_text = """from context below return text chunks relevant to the question : {question}? Context: {context} """
        
        relevance_score = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,
                                        safety_settings= {generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH}), 
                    chain_type="stuff",prompt=PromptTemplate(template=prompt_text, input_variables=["question", "context"]))
        
        result = relevance_score({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        filtered_docs = result["output_text"]

        filtered_text_chunks = get_text_chunks(filtered_docs)
        get_vector_store_for_filtered_docs(filtered_text_chunks)
        new_filtered_docs = FAISS.load_local("filtered_faiss_index",
                                             GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                                             allow_dangerous_deserialization=True)
        filtered_documents = new_filtered_docs.similarity_search(user_question)

        summary = memory

        prompt_template = """     Context:
                                  {context}?

                                  conversation summary:\n 
                                  {summary}

                                  Question: 
                                  {question}

                                  Answer:    
                          """

        chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,
                    safety_settings= {generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH}),
                              chain_type="stuff",prompt=PromptTemplate(template=prompt_template, input_variables=["question", "context", "summary"]))
       
        response = chain({"question": user_question, "input_documents": filtered_documents, "summary": summary},
                         return_only_outputs=True
                         )
        generated_answer = response["output_text"]
        print()
        print(generated_answer)
        print()
        memory.save_context({"input": user_question}, {"output": generated_answer})

        history = memory.load_memory_variables({})
        prev = f"Context: {filtered_documents} \n conversation summary: {history} \n question: {user_question}"
        summary_memory_chain.predict(input = prev)

        return generated_answer

    except IndexError as e:
        print(f"Error occurred: {e}")
        print("Safety error occurred. Skipping to next question.")
        generated_answer = "safety error"
        return generated_answer

def extract_visible_text(web_link):
    """
    Fetches the visible text content from a website using requests and BeautifulSoup.

    Args:
    web_link (str): The URL of the website.

    Returns:
        str: The extracted visible text content, or None if an error occurs.
    """

    try:
        response = requests.get(web_link)
        response.raise_for_status()  

        soup = BeautifulSoup(response.content, 'html.parser')
        visible_text = ' '.join(
            line.strip() for line in soup.find_all(string=True, recursive=True)
            if line.parent.name not in ('script', 'style', 'noscript', 'head', 'title')
            and not line.isspace()  
        )

        return visible_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching text from {web_link}: {e}")
        return None  

def main():
    st.set_page_config("PhatakaReader Pro")
    st.header("Ask me anything ~Gemini")
    csv_path = "C:\\Users\\VIBGYOR\\Desktop\\LLM-Mini-Resources\\TruthfulQA-main-BenchmarkNo.1\\TruthfulQA-main\\TruthfulQATesting_Phataka.csv"
    df = pd.read_csv(csv_path)
    question_index = 0  # Initialize question index

    for index, row in df.iterrows():
        web_link = row["Source"]
        questions = row["Question"].split("\n")
        for user_question in questions:
            domain = ""
            tone = ""  
            processed_question = preprocess_question(user_question, domain, tone)
            visible_text = ""            
            if web_link:
                visible_text = extract_visible_text(web_link)
                if visible_text:
                    question_index += 1  # Increment question index
                    output = user_input(processed_question['concatenated_string'], visible_text, "", domain, question_index)
                    if (output == "safety error"):
                        continue
                    # Write the output["response"] to the "Phatak_Answer" column in the DataFrame
                    df.loc[index, "Phatak_Answer"] = (output["response"])
                    df.loc[index, "Question_Index"] = question_index  # Assign question index
                    print("Updated DataFrame row:", df.loc[index])  # Check the updated DataFrame row

    st.write("Final DataFrame with responses:")
    st.write(df)
    
    csv_path2 = "C:\\Users\\VIBGYOR\\Desktop\\LLM-Mini-Resources\\TruthfulQA-main-BenchmarkNo.1\\TruthfulQA-main\\answers.csv"

    df.to_csv(csv_path2, index=False, mode='w')  # Open CSV file in write mode

if __name__ == "__main__":
    main()

