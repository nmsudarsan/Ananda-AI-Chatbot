
# ChatBot: Master Crypto with Ananda

Welcome to the OPIM5770-Team5/ChatBot repository.

## Introduction

The Ananda AI chatbot is a sophisticated AI-powered tool developed to assist users with cryptocurrency and DeFi-related queries. Built with cutting-edge language models and designed with both performance and scalability in mind, this chatbot bridges the gap between complex financial concepts and end-user understanding.

Our chatbot leverages state-of-the-art AI technologies to provide:
- Real-time, accurate, and context-aware responses.
- Document-based assistance using retrieval-augmented generation (RAG).
- A user-friendly interface for seamless interaction.

This README explains the setup, functionality, and technical details of the chatbot project to ensure ease of understanding and usability for both technical and non-technical users.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Folder Structure](#project-folder-structure)
3. [Technologies Used](#technologies-used)
   - [Programming Languages](#programming-languages)
   - [Frameworks and Libraries](#frameworks-and-libraries)
   - [APIs](#apis)
   - [Tools and Platforms](#tools-and-platforms)
4. [Setup Instructions](#setup-instructions)
   - [Step 1: Prerequisites](#step-1-prerequisites)
   - [Step 2: Installation](#step-2-installation)
5. [How It Works](#how-it-works)
   - [Chatbot Workflow](#chatbot-workflow)
6. [Code Explanation and File Description](#code-explanation-and-file-description)
   - [app.py](#apppy)
   - [environment.py](#environmentpy)
   - [llm_metrics_evaluation.py](#llm_metrics_evaluationpy)
   - [T-Test Analysis for LLM Comparisons.py](#t-test-analysis-for-llm-comparisonspy)
   - [Weighted average of all models.csv](#weighted-average-of-all-modelscsv)
   - [Evaluation Metrics results.csv](#evaluation-metrics-resultscsv)
   - [T-Test Results.pdf](#t-test-resultspdf)
7. [Evaluation Workflow](#evaluation-workflow)
8. [Key Features of the Chatbot](#key-features-of-the-chatbot)
9. [Crypto Trading Signals](#crypto-trading-signals)
   - [Overview](#overview)
   - [Strategies Implemented](#strategies-implemented)
   - [Key Metrics for Evaluation](#key-metrics-for-evaluation)
10. [Contact](#contact)


## Project Folder Structure

```plaintext
Chatbot/
├── app.py                           # Main application file for the chatbot
├── environment.py                   # File to load environment variables
├── .env                             # Environment configuration file
├── requirements.txt                 # Dependencies required to run the project
├── docs/                            # Folder for RAG-based document storage
├── llm_metrics_evaluation.py        # Evaluation script for assessing model performance
├── T-Test Analysis for LLM Comparisons.py # Script for T-test analysis
├── Weighted average of all models.csv     # Weighted average results for all models
├── Evaluation Metrics results.csv         # Complete evaluation metrics for all models
└── T-Test Results.pdf                     # Final T-test results for model comparisons
```

---

## Technologies Used

### Programming Languages
- **Python**: Core programming language used for all development.

### Frameworks and Libraries
- **LangChain**: Orchestrates LLMs and integrates retrieval-augmented generation (RAG) for document-based querying.
- **Streamlit**: Provides a user-friendly web interface for the chatbot.
- **FAISS (Facebook AI Similarity Search)**: Implements fast, vector-based similarity search for RAG.
- **Transformers (Hugging Face)**: Used for pre-trained embeddings and fine-tuning.
- **pandas**: Manages data analysis and manipulation.
- **scikit-learn**: Computes cosine similarity for semantic comparisons.
- **NLTK**: Calculates BLEU scores for n-gram overlaps.
- **ROUGE Score**: Assesses recall-based similarity.
- **BERTScore**: Evaluates contextual similarity between generated and reference answers.
- **dotenv**: Securely manages environment variables.

### APIs
- **Groq API**: Powers the integration with Groq LLMs (e.g., Llama3, Mixtral, and Gemma models).
- **Hugging Face API**: Retrieves pre-trained embeddings for semantic similarity.
- **LangChain API**: Enables LLM orchestration and tracking.

### Tools and Platforms
- **Microsoft Excel/Google Sheets**: For tabular data organization and analysis.
- **Streamlit Hosting**: Provides a web interface for deploying the chatbot.
- **FAISS**: Optimizes document retrieval for RAG-based queries.
- **Git**: For version control and collaborative development.
- **Jupyter Notebook**: Used for exploratory data analysis and prototyping.


## Setup Instructions

### Step 1: Prerequisites
Before starting, ensure the following:
- Python 3.8 or higher installed on your system.
- A virtual environment tool like `venv` or `conda`.
- API keys for Groq, LangChain, and Hugging Face.

### Step 2: Installation

#### Clone the Repository:
```bash
git clone https://github.com/your-repo/ananda-chatbot.git
cd ananda-chatbot
```

#### Install Dependencies:
Use the following command to install all required Python libraries:
```bash
pip install -r requirements.txt
```

#### Configure API Keys:
Create a `.env` file in the root directory and add your API keys:
```plaintext
LANGCHAIN_API_KEY=your_langchain_api_key
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
LANGCHAIN_PROJECT=your_project_name
```

#### Add Documents (Optional):
If using the RAG configuration, add your PDF files to the `docs/` folder. These files will be processed and used to answer document-specific questions.

#### Run the Application:
Launch the chatbot using Streamlit:
```bash
streamlit run app.py
```
Open the provided URL in your browser to interact with the chatbot.

---

## How It Works

### Chatbot Workflow
1. **User Interaction**: The user inputs a query through a simple and interactive chat interface.
2. **Base or RAG Configuration**:
   - If documents are uploaded to the `docs/` folder, the chatbot retrieves relevant content using FAISS and Hugging Face embeddings.
   - If no documents are available, the chatbot relies solely on the Llama 3 model to generate responses.
3. **Response Generation**: The chatbot processes the query and provides an answer based on context.


## Code Explanation and File Description

### app.py
**Purpose**: This is the main script to launch the chatbot. It integrates user interaction, LLM (Large Language Model) responses, and optional document retrieval for RAG (Retrieval-Augmented Generation)-based queries.

### environment.py
**Purpose**: Loads environment variables required for the chatbot's configuration and API access.

### llm_metrics_evaluation.py
**Purpose**: Evaluates the performance of three LLMs (Llama, Mixtral, and Gemma) using various metrics.

### T-Test Analysis for LLM Comparisons.py
**Purpose**: Performs statistical tests (paired T-tests) to determine significant differences between the models.

### Weighted average of all models.csv
**Purpose**: Contains the weighted average scores of all models for 96 questions.

### Evaluation Metrics results.csv
**Purpose**: Stores detailed metrics (BLEU, ROUGE-L, Cosine Similarity, BERTScore, Accuracy) for each model and question.

### T-Test Results.pdf
**Purpose**: Summarizes the results of all model comparisons with tables for:
- T-statistic values.
- P-values.
- Statistical significance.

---

## Evaluation Workflow

1. **Chatbot Setup:**
   - Launch `app.py` to initialize the chatbot.
   - Interact with the chatbot and analyze responses.

2. **Model Evaluation:**
   - Use `llm_metrics_evaluation.py` to compute performance metrics.
   - Save results to `comparison_results_weighted_RAG.csv`.

3. **Statistical Analysis:**
   - Run `T-Test Analysis for LLM Comparisons.py` for detailed comparisons.
   - Review `T-Test Results.pdf` for insights.


## Key Features of the Chatbot

- **LLM-Driven Responses:** Powered by Groq's Llama3, optimized for cryptocurrency-related queries.
- **Document Retrieval:** Incorporates RAG-based techniques for improved context and reliability.
- **User-Friendly Interface:** Interactive UI built with Streamlit for smooth communication.
- **Evaluation Metrics:** Robust system for assessing model performance using BLEU, ROUGE, and BERTScore.

---

## Crypto Trading Signals

### Overview

This project analyzes different trading strategies (such as Simple Moving Average (SMA), Exponential Moving Average (EMA), MACD, RSI, and Bollinger Bands) applied to historical Bitcoin (BTC) and Ethereum (ETH) data. The goal is to evaluate the performance of these strategies using metrics like Sharpe Ratio, Annualized Volatility, Annualized Return, and Strategy vs Market Returns. The Sharpe Ratio is the primary metric used to rank the effectiveness of each strategy.

### Strategies Implemented

- **Moving Averages (SMA and EMA):** These strategies compare short-term and long-term averages to determine buy/sell signals.
- **RSI (Relative Strength Index):** Uses momentum indicators to assess whether an asset is overbought or oversold.
- **MACD (Moving Average Convergence Divergence):** Identifies changes in the strength, direction, momentum, and duration of a trend.
- **Bollinger Bands:** Measures volatility and overbought/oversold conditions with standard deviations.

### Key Metrics for Evaluation

- **Sharpe Ratio:** Measures risk-adjusted return.
- **Annualized Volatility:** Assesses how much the strategy's value fluctuates yearly.
- **Annualized Return:** The compounded annual return from the strategy.
- **Strategy vs Market Return:** Compares strategy returns to overall market performance.


## Contact

For questions, suggestions, or collaboration opportunities:  
- **Email**: team5@opim5770.com  
- **GitHub Repository**: [Ananda Chatbot](https://github.com/your-repo/OPIM5770-Team5)
