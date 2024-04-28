import csv
import nltk.translate.bleu_score as bleu
from rouge import Rouge
import tensorflow as tf
import tensorflow_hub as hub
import chardet 
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import jaccard_distance

# Load the BERT module
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(module_url)

# Function to calculate BERT score
def bert_score(hypothesis, reference):
    # Compute embeddings for hypothesis and reference
    hypothesis_embedding = embed([hypothesis])[0]
    reference_embedding = embed([reference])[0]
    
    # Compute cosine similarity between the embeddings
    similarity = cosine_similarity([hypothesis_embedding], [reference_embedding])[0][0]
    return similarity

# Function to calculate Jaccard similarity
def jaccard_similarity(hypothesis, reference):
    hypothesis_tokens = set(hypothesis.split())
    reference_tokens = set(reference.split())
    return 1 - jaccard_distance(hypothesis_tokens, reference_tokens)

# Function to calculate Moore Index
def moore_index(hypothesis, reference):
    common_words = set(hypothesis.split()) & set(reference.split())
    return len(common_words) / len(reference.split())

# Function to calculate F1 Score
def f1_score(hypothesis, reference):
    common_words = set(hypothesis.split()) & set(reference.split())
    precision = (len(common_words)+1) / (len(hypothesis.split())+1)
    recall = len(common_words) / len(reference.split())
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

# Lists to store Phatak_Answer and Correct Answers
phatak_answers = []
correct_answers = []

# Read CSV and extract data
csv_file = r'C:\Users\VIBGYOR\Desktop\LLM-Mini-Resources\TruthfulQA-main-BenchmarkNo.1\TruthfulQA-main\final_compliation_METRICS - Copy.csv'
# csv_file = r"C:\Users\VIBGYOR\Documents\Downloads\Final_Compilation_Copy_Sorted.csv"

# Detect encoding using chardet
rawdata = open(csv_file, 'rb').read()
result = chardet.detect(rawdata)
encoding = result['encoding']

with open(csv_file, 'r', encoding=encoding) as file:  # Use detected encoding
  reader = csv.DictReader(file)
  next(reader)  # Skip header row
  for row in reader:
    phatak_answers.append(row['Phatak_Answer'])
    correct_answers.append(row['Correct Answers'])

# Lists to store metric values
bleu_scores = []
rouge_scores = []
bleurt_scores = []
moore_indices = [] 
f1_scores = []
bert_scores = []
jaccard_similarities = []


for phatak_answer, correct_answer in zip(phatak_answers, correct_answers):
    
    # Moore Index
    moore_idx = moore_index(phatak_answer, correct_answer)
    moore_indices.append(moore_idx)

    # F1 Score
    f1 = f1_score(phatak_answer, correct_answer)
    f1_scores.append(f1)
    
    # BLEU Score (using bigram - BLEU-2)
    bleu_score = bleu.sentence_bleu([correct_answer.split()], phatak_answer.split(),smoothing_function=SmoothingFunction().method5)
    bleu_scores.append(bleu_score)
    
    # BERT Score
    bert = bert_score(phatak_answer, correct_answer)
    bert_scores.append(bert)
    
    # Jaccard Similarity
    jaccard_sim = jaccard_similarity(phatak_answer, correct_answer)
    jaccard_similarities.append(jaccard_sim)

for score in enumerate(f1_scores):
    print(score)

# Calculate average values
avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
avg_moore_idx = sum(moore_indices) / len(moore_indices)
avg_f1_score = sum(f1_scores) / len(f1_scores)
avg_bert_score = np.mean(bert_scores)
avg_jaccard_similarity = np.mean(jaccard_similarities)

# Print average values
print("Average BLEU Score:", avg_bleu_score)
print("Average Moore Index:", avg_moore_idx)
print("Average F1 Score:", avg_f1_score)
print("Average BERT Score:", avg_bert_score)
print("Average Jaccard Similarity:", avg_jaccard_similarity)
