import pandas as pd  # Library for data manipulation and analysis
import re  # Library for regular expressions (used for extracting question numbers)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # For calculating BLEU scores
from sklearn.metrics.pairwise import cosine_similarity  # For calculating cosine similarity between embeddings
from rouge_score import rouge_scorer  # For calculating ROUGE scores (recall-based similarity)
from bert_score import score as bert_score  # For BERT-based similarity scoring
from transformers import AutoTokenizer, AutoModel  # For using pre-trained transformer models for embeddings
import torch  # Library for tensor operations
import string  # For string manipulation

# File paths to the CSV files containing gold-standard answers and model-generated answers
gold_standard_path = '/Users/dharun/Desktop/Ananda/formatted_gold_answers.csv'
llama_answers_path = '/Users/dharun/Desktop/Ananda/formatted_llama_answers.csv'
Mixtral_answers_path = '/Users/dharun/Desktop/Ananda/formatted_Mixtral_answers.csv'
gemma_answers_path = '/Users/dharun/Desktop/Ananda/formatted_gemma_answers.csv'
output_file = '/Users/dharun/Desktop/Ananda/comparison_results_weighted_RAG.csv'

# Load the CSV files into pandas DataFrames for easy data manipulation
gold_standard_data = pd.read_csv(gold_standard_path)  # Gold-standard answers
llama_data = pd.read_csv(llama_answers_path)  # Answers from Llama model
Mixtral_data = pd.read_csv(Mixtral_answers_path)  # Answers from Mixtral model
gemma_data = pd.read_csv(gemma_answers_path)  # Answers from Gemma model

# Rename columns for consistency and better understanding
gold_standard_data.columns = ["Question", "Gold Answer"]
llama_data.columns = ["Question", "Llama Answer"]
Mixtral_data.columns = ["Question", "Mixtral Answer"]
gemma_data.columns = ["Question", "Gemma Answer"]

# Function to extract numeric question numbers from the "Question" column
# This ensures proper alignment of questions across different datasets
def extract_question_number(question_text):
    match = re.search(r'\d+', question_text)  # Find the first numeric sequence in the question text
    return int(match.group()) if match else None  # Return the number if found, else None

# Add a new "Question Number" column to each DataFrame using the extract_question_number function
gold_standard_data['Question Number'] = gold_standard_data['Question'].apply(extract_question_number)
llama_data['Question Number'] = llama_data['Question'].apply(extract_question_number)
Mixtral_data['Question Number'] = Mixtral_data['Question'].apply(extract_question_number)
gemma_data['Question Number'] = gemma_data['Question'].apply(extract_question_number)

# Merge all the datasets on the "Question Number" column
# This ensures that answers from different models and the gold standard align by question
merged_df = pd.merge(gold_standard_data, llama_data, on="Question Number", how="left", suffixes=('_gold', '_llama'))
merged_df = pd.merge(merged_df, Mixtral_data, on="Question Number", how="left", suffixes=('', '_Mixtral'))
merged_df = pd.merge(merged_df, gemma_data, on="Question Number", how="left", suffixes=('', '_gemma'))

# Initialize tokenizer and model for generating embeddings for BERTScore and cosine similarity
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ROUGE scorer for calculating ROUGE-L scores
rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Smoothing function to handle BLEU score calculation for short sequences
smoothing = SmoothingFunction()

# Define weights for each metric to calculate a weighted average score
weights = {
    "BLEU": 0.20,  # Weight for BLEU score
    "ROUGE-L": 0.20,  # Weight for ROUGE-L score
    "Cosine Similarity": 0.25,  # Weight for cosine similarity
    "BERTScore": 0.25,  # Weight for BERTScore
    "Accuracy": 0.10  # Weight for accuracy
}

# Define a threshold for accuracy based on cosine similarity
# If cosine similarity is above this threshold, we consider the answer accurate
accuracy_threshold = 0.7

# Function to calculate embeddings for a given text input
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)  # Tokenize input text
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Generate embeddings by averaging token embeddings
    return embeddings

# Function to clean and normalize text for consistent metric calculations
def normalize_text(text):
    if pd.isna(text):  # Check if text is NaN (missing)
        return ""
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Initialize a list to store metrics for each question
metrics_data = []

# Loop through each row in the merged DataFrame to calculate metrics
for _, row in merged_df.iterrows():
    # Normalize answers to ensure consistency in metric calculations
    gold_answer = normalize_text(row["Gold Answer"])
    llama_answer = normalize_text(row["Llama Answer"])
    Mixtral_answer = normalize_text(row["Mixtral Answer"])
    gemma_answer = normalize_text(row["Gemma Answer"])
    question = row["Question_gold"]  # Store the original question text for reference
    
    # Dictionary to store metrics for the current question
    row_metrics = {
        "Question": question,
        "Gold Answer": row["Gold Answer"],  # Original gold-standard answer
        "Llama Answer": row["Llama Answer"],
        "Mixtral Answer": row["Mixtral Answer"],
        "Gemma Answer": row["Gemma Answer"]
    }

    # Function to calculate metrics for a given model's answer
    def calculate_metrics(model_answer, model_name):
        # Handle empty answers
        if model_answer.strip() == "":
            return {
                f"{model_name} BLEU": 0,
                f"{model_name} Cosine Similarity": 0,
                f"{model_name} ROUGE-L": 0,
                f"{model_name} BERTScore": 0,
                f"{model_name} Accuracy": 0,
                f"{model_name} Weighted Average": 0  # Assign zero if the answer is empty
            }

        # Calculate BLEU score
        bleu_score = sentence_bleu([gold_answer.split()], model_answer.split(), smoothing_function=smoothing.method1)
        
        # Calculate cosine similarity between embeddings
        gold_embedding = get_embeddings(gold_answer)
        model_embedding = get_embeddings(model_answer)
        cosine_score = cosine_similarity(gold_embedding.detach().numpy(), model_embedding.detach().numpy())[0][0]
        
        # Calculate ROUGE-L score
        rouge_score = rouge_scorer.score(gold_answer, model_answer)['rougeL'].fmeasure
        
        # Calculate BERTScore
        _, _, bert_f1 = bert_score([model_answer], [gold_answer], lang="en", rescale_with_baseline=True)
        bert_score_val = bert_f1.mean().item()
        
        # Determine accuracy based on cosine similarity threshold
        accuracy = 1 if cosine_score >= accuracy_threshold else 0

        # Calculate the weighted average of all metrics
        weighted_avg = (bleu_score * weights["BLEU"] +
                        cosine_score * weights["Cosine Similarity"] +
                        rouge_score * weights["ROUGE-L"] +
                        bert_score_val * weights["BERTScore"] +
                        accuracy * weights["Accuracy"])

        # Return all calculated metrics for the model
        return {
            f"{model_name} BLEU": bleu_score,
            f"{model_name} Cosine Similarity": cosine_score,
            f"{model_name} ROUGE-L": rouge_score,
            f"{model_name} BERTScore": bert_score_val,
            f"{model_name} Accuracy": accuracy,
            f"{model_name} Weighted Average": weighted_avg
        }

    # Calculate metrics for all three models
    row_metrics.update(calculate_metrics(llama_answer, "Llama"))
    row_metrics.update(calculate_metrics(Mixtral_answer, "Mixtral"))
    row_metrics.update(calculate_metrics(gemma_answer, "Gemma"))

    # Append the calculated metrics for the current question
    metrics_data.append(row_metrics)

# Convert the list of metrics into a DataFrame
results_df = pd.DataFrame(metrics_data)

# Calculate average metrics for each model across all questions
average_metrics = {}
for model_name in ["Llama", "Mixtral", "Gemma"]:
    average_metrics[f"{model_name} Average BLEU"] = results_df[f"{model_name} BLEU"].mean()
    average_metrics[f"{model_name} Average Cosine Similarity"] = results_df[f"{model_name} Cosine Similarity"].mean()
    average_metrics[f"{model_name} Average ROUGE-L"] = results_df[f"{model_name} ROUGE-L"].mean()
    average_metrics[f"{model_name} Average BERTScore"] = results_df[f"{model_name} BERTScore"].mean()
    average_metrics[f"{model_name} Average Accuracy"]

# Print average metrics for each model to determine the best performer
for metric, value in average_metrics.items():
    print(f"{metric}: {value}")

# Save detailed metrics for all questions to a CSV file
results_df.to_csv(output_file, index=False)
print(f"Comparison results saved at: {output_file}")