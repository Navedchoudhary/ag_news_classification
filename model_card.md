# Model Card: News Article Classification

## Overview
This project involves **News Article Classification** using two different approaches:
1. **Naive Bayes Classifier with TF-IDF**
2. **BERT (Transformer-based Model)**

The goal is to classify news articles into four categories:
- **Sports (rec.sport.baseball)**
- **Medical (sci.med)**
- **Politics (talk.politics.mideast)**
- **Technology (comp.graphics)**

## Model Details

### 1. Naive Bayes Classifier (TF-IDF)
- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency)
- **Strengths:** Fast, interpretable, and effective for text classification
- **Weaknesses:** Assumes word independence, struggles with complex semantics

### 2. BERT Model (Transformer-based)
- **Architecture:** `bert-base-uncased` (Pretrained Transformer)
- **Fine-Tuned On:** The extracted dataset (subset of `20 Newsgroups`)
- **Strengths:** Captures deep contextual understanding and performs well on complex text
- **Weaknesses:** Computationally expensive, requires significant memory

## Dataset
- **Source:** `fetch_20newsgroups` from `scikit-learn`
- **Categories Used:**
  - `rec.sport.baseball`
  - `sci.med`
  - `talk.politics.mideast`
  - `comp.graphics`
- **Size:** ~3,000 articles (split into training & testing sets)
- **Preprocessing Steps:**
  - Removing stopwords & special characters
  - Tokenization & lemmatization (for Naive Bayes)
  - BERT tokenization (for Transformer-based model)

## Evaluation Metrics

| Model        | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|------------|--------|---------|
| Naive Bayes | ~85%    | Moderate  | Moderate | Moderate |
| BERT        | ~92%    | High      | High    | High    |

## Limitations
- **Data Imbalance:** Some categories may have fewer samples, impacting classification.
- **Computational Constraints:** BERT requires high GPU resources for training and inference.
- **Domain Adaptation:** Models trained on this dataset may not generalize well to other news domains.

## Future Improvements
- **Hyperparameter tuning** for better model performance.
- **Incorporate LSTMs or CNNs** for alternative deep learning-based classification.
- **Experiment with DistilBERT or ALBERT** for a more lightweight transformer model.
- **Deploy as an API** to classify live news articles in real-time.

## Usage
- The models can be loaded using `joblib` (for Naive Bayes) and `transformers` (for BERT).
- The trained models can be deployed using FastAPI or Flask for real-world applications.

## License
MIT License - Open for research and educational use.

---

Let me know if you'd like any modifications! 🚀
