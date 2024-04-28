import csv
import nltk.translate.bleu_score as bleu
import sacrebleu
# from rouge import Rouge
import tensorflow as tf
import tensorflow_hub as hub
import chardet 
from nltk.translate.bleu_score import SmoothingFunction

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
# rouge_scores = []
bleurt_scores = []
moore_indices = []
f1_scores = []
sacrebleu_scores = []

count = 0

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
    
    # # Rouge
    # rouge = Rouge()
    # rouge_scores = rouge.get_scores(phatak_answer, correct_answer)
    # # Handle cases where Rouge scores are unavailable for short sentences
    # if not rouge_scores:
    #     # Assign a default value of 0 for Rouge-L F1 score
    #     rouge_scores = [{'rouge-l': {'f': 0.0}}]
    # rouge_score = rouge_scores[0]['rouge-l']['f']

    # SACREBLEU Score
    sacrebleu_score = sacrebleu.raw_corpus_bleu([phatak_answer], [[correct_answer]], .01).score
    sacrebleu_scores.append(sacrebleu_score)
    
    print("Question num: ", count+1)
    
    print("moore_idx ", moore_idx)
    print("f1 ", f1)
    print("bleu_score", bleu_score)
    print("sacrebleu_score", sacrebleu_score)
    print()

# Calculate average values
avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
avg_moore_idx = sum(moore_indices) / len(moore_indices)
avg_f1_score = sum(f1_scores) / len(f1_scores)
# avg_rouge_score = sum(rouge_score) / len(rouge_scores)
avg_sacrebleu_score = sum(sacrebleu_scores) / len(sacrebleu_scores)

# Print average values
print("Average BLEU Score:", avg_bleu_score)
print("Average Moore Index:", avg_moore_idx)
print("Average F1 Score:", avg_f1_score)
# print("Average Rouge Score:", avg_rouge_score)
print("Average sacreBLEU Score:", avg_sacrebleu_score)
