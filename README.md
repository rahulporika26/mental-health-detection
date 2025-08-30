🧠 Mental Health Detection Using NLP

This repository explores multiple machine learning and deep learning approaches for multi-class mental health status detection based on user-generated text data.

Our models classify text into 7 categories:

Anxiety
Bipolar
Depression
Normal
Personality Disorder
Stress
Suicidal
📂 Repository Contents

DistilBERT_model_91%.ipynb → Fine-tuned DistilBERT model achieving 91% accuracy
biLSTM+pretrained_word2vec_model.ipynb → BiLSTM with pretrained Word2Vec embeddings
An_Ensemble_Machine_Learning_Pipeline.ipynb → Hybrid ML + DL ensemble approach
balanced_dataset_clean_zz.csv → Processed & balanced dataset
Original_dataset.csv → Raw dataset (before preprocessing)
📊 Dataset

The dataset consists of user-generated mental health statements labeled into 7 categories.

Preprocessing steps:

Removed stopwords, URLs, and special characters
Balanced classes to handle class imbalance
Tokenized and padded sequences for deep learning models
🔍 Exploratory Data Analysis (EDA)

We conducted:

Class distribution analysis (bar & pie charts)
Word clouds for each category
Text length distribution
Most frequent words per class
🧪 Models Implemented

1️⃣ DistilBERT (Transformers)

Tokenizer: DistilBertTokenizerFast

Model: TFDistilBertForSequenceClassification

Optimizer: AdamW + learning rate scheduler

Performance:

Accuracy: 91.06%
Strong across most classes, slight dip for Depression & Suicidal
2️⃣ BiLSTM + Pretrained Word2Vec

Embedding: Pretrained Google News Word2Vec
Architecture: Bi-directional LSTM with dropout
Output: Dense softmax classification layer
3️⃣ Ensemble Machine Learning

Models used: Logistic Regression, Random Forest, XGBoost
Ensemble method: Soft voting (probability-based)
Benefit: Improved macro-average F1 score
📈 Results

Class	Precision	Recall	F1-score
Anxiety	0.96	0.97	0.97
Bipolar	0.97	0.98	0.98
Depression	0.76	0.72	0.74
Normal	0.95	0.95	0.95
Personality Disorder	0.99	0.98	0.98
Stress	0.95	0.97	0.96
Suicidal	0.78	0.80	0.79
Overall Performance:

✅ Accuracy: 91.06%
✅ Macro F1-score: 0.91
