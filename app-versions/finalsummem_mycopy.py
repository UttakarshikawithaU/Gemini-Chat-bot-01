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
from langchain_core import BaseLanguageModel, BaseChatMessageHistory, BaseChatMessage
from langchain.memory.summary import ConversationSummaryMemory
from langchain.chains import ConversationChain


def load_dotenv():
    """Loads environment variables from a .env file."""
    try:
        from pathlib import Path
        env_path = Path(".") / ".env"
        if env_path.exists():
            st.info("Loading environment variables from .env file.")
            with env_path.open(encoding="utf-8") as f:
                for line in f:
                    key, value = line.strip().split("=")
                    os.environ[key] = value
    except Exception as e:
        st.warning("Error loading environment variables: %s" % e)


def preprocess_question(user_question, domain):
    """Preprocesses the user question by concatenating it with priming and style instructions."""
    priming_variable = f"You are an expert that provides in-detail answers with reference to the questions in the {domain} domain."
    style_and_tone_variable = "Use 10th-grade language and explain things in a simple way that anyone can understand."

    processed_question = {
        "concatenated_string": f"{user_question} {priming_variable} {style_and_tone_variable}."
    }

    return processed_question


def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Splits text into chunks of a specific size with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Creates a vector store for text chunks using Google's generative AI embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_vector_store_for_filtered_docs(text_chunks):
    """Creates a separate vector store for filtered text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("filtered_faiss_index")
    return vector_store


def compute_jaccard_similarity(set1, set2):
    """Calculates Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def compute_accuracy_jaccard(generated_answer, ground_truth_answer):
    """Calculates Jaccard similarity accuracy between generated and ground truth answers."""
    tokens_generated = set(word_tokenize(generated_answer.lower()))
    tokens_ground_truth = set(word_tokenize(ground_truth_answer.lower()))
    jaccard_similarity = compute_jaccard_similarity(tokens_generated, tokens_ground_truth)
    return jaccard_similarity

def compute_bleu_score(reference, hypothesis):
    """Calculates BLEU score between the reference and hypothesis."""
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    return sentence_bleu([reference_tokens], hypothesis_tokens)


def compute_meteor_score(reference, hypothesis):
    """Calculates METEOR score between the reference and hypothesis."""
    reference_tokens = word_tokenize(reference)  # Tokenize the reference here
    hypothesis_tokens = word_tokenize(hypothesis)
    score = meteor_score([reference_tokens], hypothesis_tokens)
    return score


def compute_rouge_scores(reference, hypothesis):
    """Calculates ROUGE scores between the reference and hypothesis."""
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores  # Assuming a single reference


def compute_f1_score(reference, hypothesis):
    """Calculates F1 score between the reference and hypothesis."""
    reference_tokens = set(word_tokenize(reference.lower()))
    hypothesis_tokens = set(word_tokenize(hypothesis.lower()))
    intersection = len(reference_tokens.intersection(hypothesis_tokens))
    precision = intersection / len(hypothesis_tokens) if len(hypothesis_tokens) > 0 else 0
    recall = intersection / len(reference_tokens) if len(reference_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def generate_output(generated_answer, metrics):
    """Generates a dictionary containing the response, metrics, and bullet points for better presentation."""
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


def user_input(user_question, pdf_docs, ground_truth_answer, use_gemini_knowledge, domain):
    """Processes the user's question, retrieves relevant information from PDFs, and generates a response."""

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    memory = ConversationSummaryMemory(llm=llm, max_total_history_tokens=400)  # 10 messages

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    prompt_text = """from context below return text chunks relevant to the question : {question}? Context: {context} """

    relevance_score = load_qa_chain(
        ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3),
        chain_type="stuff",
        prompt=PromptTemplate(template=prompt_text, input_variables=["question", "context"]),
    )
    result = relevance_score({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    filtered_docs = result["output_text"]

    st.write("Filtered Doc: ", filtered_docs)

    filtered_text_chunks = get_text_chunks(filtered_docs)
    get_vector_store_for_filtered_docs(filtered_text_chunks)
    new_filtered_docs = FAISS.load_local(
        "filtered_faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True
    )
    filtered_documents = new_filtered_docs.similarity_search(user_question)

    summary = memory

    prompt_template = """
        Context:
            {context}?

        conversation summary:
            {summary}

        Question:
            {question}

        Answer:
    """

    chain = load_qa_chain(
        ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7),
        chain_type="stuff",
        prompt=PromptTemplate(template=prompt_template, input_variables=["question", "context", "summary"]),
    )

    # Pass documents as input to LangChain
    response = chain(
        {"question": user_question, "input_documents": filtered_documents, "summary": summary}, return_only_outputs=True
    )
    generated_answer = response["output_text"]

    history = memory.load_memory_variables({})
    prev = f"Context: {filtered_documents} \n conversation summary: {history} \n question: {user_question}"
    summary_memory.predict(input=prev)

    accuracy_jaccard = compute_accuracy_jaccard(generated_answer, ground_truth_answer)
    bleu_score = compute_bleu_score(ground_truth_answer, generated_answer)
    meteor_score = compute_meteor_score(ground_truth_answer, generated_answer)
    rouge_scores = compute_rouge_scores(ground_truth_answer, generated_answer)
    f1_score_value = compute_f1_score(ground_truth_answer, generated_answer)

    output, bullet_points, json_format = generate_output(
        generated_answer,
        {
            "accuracy_jaccard": accuracy_jaccard,
            "bleu_score": bleu_score,
            "meteor_score": meteor_score,
            "rouge_scores": rouge_scores,
            "f1_score": f1_score_value,
        },
    )

    # st.write("Processed Question:")
    st.write("Response:", output["response"])
    st.write("Priming:", output.get("priming", ""))
    st.write("Style and Tone Instructions:", output.get("style_and_tone_instructions", ""))

    st.success("Evaluation Metrics:")
    for point in bullet_points:
        st.success(f"- {point}")

    st.success("Evaluation Metrics (JSON Format):")
    st.success(json_format)

    return output


def main():
    """Sets up the Streamlit application and calls the user_input function."""
    st.set_page_config("PhatakaReader Pro")
    st.header("Ask me anything ~Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")
    domain = st.text_input("Enter the domain (e.g., plumbing, history)")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        ground_truth_answer = st.text_input("Enter Ground Truth Answer for Evaluation")
        use_gemini_knowledge = st.checkbox("Use Gemini only :)")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if user_question and pdf_docs and ground_truth_answer and domain:
                    processed_question = preprocess_question(user_question, domain)
                    st.write("Preprocessed question: ",processed_question)
                    output = user_input(processed_question['concatenated_string'], pdf_docs, ground_truth_answer, use_gemini_knowledge, domain)
                    st.write("Response:", output["response"])                

                else:
                    st.warning("Please provide input for all fields")
                    
    # st.write("Response:", output["response"])                

if __name__ == "__main__":
    main()
