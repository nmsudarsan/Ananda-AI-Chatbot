import pandas as pd
from scipy.stats import ttest_rel

# Load the CSV file containing the results of models for all comparisons
# The CSV should have columns like Question ID and scores for different models
data = pd.read_csv('/Users/dharun/Desktop/Ananda/results/T-test.csv')

# Function to perform a paired t-test for two sets of scores
# A paired t-test checks whether the means of two related groups (e.g., Base and RAG scores for the same questions) are significantly different
def perform_t_test(model1_scores, model2_scores):
    # ttest_rel computes the T-statistic and P-value
    t_stat, p_val = ttest_rel(model1_scores, model2_scores)
    return t_stat, p_val

# Extracting key data from the CSV file
# Question IDs help us align the questions across models
question_ids = data['Question ID']  # This column contains IDs of the questions used for comparison

# Organizing scores of different models into a dictionary for easier access
models = {
    'Base_Llama': data['Base_model_Llama'],  # Scores for the Base Llama model
    'Base_Mixtral': data['base_model_Mixtral'],  # Scores for the Base Mixtral model
    'Base_Gemma': data['Base_Model_Gemma2'],  # Scores for the Base Gemma model
    'RAG_Llama': data['Rag_Model_Llama'],  # Scores for the RAG-enhanced Llama model
    'RAG_Mixtral': data['Rag_Model_Mixtral'],  # Scores for the RAG-enhanced Mixtral model
    'RAG_Gemma': data['Rag_Model_Gemma2']  # Scores for the RAG-enhanced Gemma model
}

# Helper function to determine if the RAG model shows improvement over the Base model
# Compares the average scores of the two models and returns "Yes" if RAG outperforms Base
def check_improvement(base_scores, rag_scores):
    base_avg = base_scores.mean()  # Calculate the average score for the Base model
    rag_avg = rag_scores.mean()  # Calculate the average score for the RAG model
    return "Yes" if rag_avg > base_avg else "No"  # Improvement exists if RAG's average is higher

# Performing paired t-tests for Base model comparisons
print("\nBase Model Comparisons:")
# Loop through each pair of Base models to compare them
for (model1_name, model2_name) in [
    ('Base_Llama', 'Base_Mixtral'), 
    ('Base_Llama', 'Base_Gemma'), 
    ('Base_Mixtral', 'Base_Gemma')
]:
    # Extract scores for the two models being compared
    model1_scores = models[model1_name]
    model2_scores = models[model2_name]
    
    # Calculate the average scores for the two models
    model1_avg = model1_scores.mean()
    model2_avg = model2_scores.mean()
    
    # Perform a paired t-test between the two models
    t_stat, p_val = perform_t_test(model1_scores, model2_scores)
    
    # Check if the P-value indicates statistical significance (P < 0.05)
    significant = "Yes" if p_val < 0.05 else "No"
    
    # Identify which model has the higher average score
    best_model = model1_name if model1_avg > model2_avg else model2_name

    # Print the results for the current comparison
    print(f"Comparison: {model1_name} vs. {model2_name}")
    print(f"Model 1 Avg. Weighted Score: {model1_avg}")
    print(f"Model 2 Avg. Weighted Score: {model2_avg}")
    print(f"T-Statistic: {t_stat}")  # Indicates the magnitude and direction of the difference
    print(f"P-Value: {p_val}")  # Shows the statistical significance
    print(f"Significant: {significant}")
    print(f"Best Model: {best_model}\n")

# Performing paired t-tests for RAG model comparisons
print("\nRAG Model Comparisons:")
# Compare RAG versions of the models in pairs
for (model1_name, model2_name) in [
    ('RAG_Llama', 'RAG_Mixtral'), 
    ('RAG_Llama', 'RAG_Gemma'), 
    ('RAG_Mixtral', 'RAG_Gemma')
]:
    model1_scores = models[model1_name]
    model2_scores = models[model2_name]
    
    model1_avg = model1_scores.mean()  # Calculate the average score for Model 1
    model2_avg = model2_scores.mean()  # Calculate the average score for Model 2
    
    t_stat, p_val = perform_t_test(model1_scores, model2_scores)  # Perform a t-test
    significant = "Yes" if p_val < 0.05 else "No"  # Determine significance
    best_model = model1_name if model1_avg > model2_avg else model2_name  # Identify the better model

    # Print the results for the current comparison
    print(f"Comparison: {model1_name} vs. {model2_name}")
    print(f"Model 1 Avg. Weighted Score: {model1_avg}")
    print(f"Model 2 Avg. Weighted Score: {model2_avg}")
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_val}")
    print(f"Significant: {significant}")
    print(f"Best Model: {best_model}\n")

# Comparing each Base model with its corresponding RAG model
print("\nBase vs. RAG Model Comparisons:")
for (base_name, rag_name) in [
    ('Base_Llama', 'RAG_Llama'), 
    ('Base_Mixtral', 'RAG_Mixtral'), 
    ('Base_Gemma', 'RAG_Gemma')
]:
    base_scores = models[base_name]
    rag_scores = models[rag_name]
    
    base_avg = base_scores.mean()  # Average score of the Base model
    rag_avg = rag_scores.mean()  # Average score of the RAG model
    
    t_stat, p_val = perform_t_test(base_scores, rag_scores)  # Perform a t-test
    significant = "Yes" if p_val < 0.05 else "No"  # Determine significance
    improvement = check_improvement(base_scores, rag_scores)  # Check for improvement with RAG
    best_model = base_name if base_avg > rag_avg else rag_name  # Identify the better model

    # Print the results for the Base vs. RAG comparison
    print(f"Comparison: {base_name} vs. {rag_name}")
    print(f"Base Avg. Weighted Score: {base_avg}")
    print(f"RAG Avg. Weighted Score: {rag_avg}")
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_val}")
    print(f"Significant: {significant}")
    print(f"Improvement with RAG: {improvement}")
    print(f"Best Model: {best_model}\n")

# Identifying the top 3 best models based on P-values
print("\nTop 3 Best Models Overall (Based on P-Value):")
# Combine all comparisons into a single list for ranking
all_comparisons = [
    ('Llama Base vs. Mixtral Base', models['Base_Llama'], models['base_model_Mixtral']),
    ('Llama Base vs. Gemma Base', models['Base_Llama'], models['Base_Model_Gemma2']),
    ('Mixtral Base vs. Gemma Base', models['base_model_Mixtral'], models['Base_Model_Gemma2']),
    ('Llama RAG vs. Mixtral RAG', models['Rag_Model_Llama'], models['Rag_Model_Mixtral']),
    ('Llama RAG vs. Gemma RAG', models['Rag_Model_Llama'], models['Rag_Model_Gemma2']),
    ('Mixtral RAG vs. Gemma RAG', models['Rag_Model_Mixtral'], models['Rag_Model_Gemma2']),
    ('Llama Base vs. Llama RAG', models['Base_Llama'], models['Rag_Model_Llama']),
    ('Mixtral Base vs. Mixtral RAG', models['base_model_Mixtral'], models['Rag_Model_Mixtral']),
    ('Gemma Base vs. Gemma RAG', models['Base_Model_Gemma2'], models['Rag_Model_Gemma2'])
]

# Calculate P-values for all comparisons and sort by significance
top_3 = sorted(
    [(name, perform_t_test(model1, model2)[1]) for name, model1, model2 in all_comparisons],
    key=lambda x: x[1]
)[:3]

# Print the top 3 comparisons with the lowest P-values
for comparison, p_val in top_3:
    print(f"Comparison: {comparison}")
    print(f"P-Value: {p_val}\n")
