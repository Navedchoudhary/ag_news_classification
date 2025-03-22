import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import torch
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load AG News dataset (similar dataset, as AG News isn't directly in sklearn)
from sklearn.datasets import fetch_20newsgroups
categories = ['rec.sport.baseball', 'sci.med', 'talk.politics.mideast', 'comp.graphics']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
# data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'), download_if_missing=False)

df = pd.DataFrame({'text': data.data, 'category': data.target})

# Data Preprocessing
# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Function to clean and preprocess text data."""
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    tokens = word_tokenize(text.lower())  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Data Visualization
plt.figure(figsize=(8,5))
sns.countplot(x=df['category'])
plt.title("Category Distribution")
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_text']))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['category'], test_size=0.2, random_state=42)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes Model Training and Evaluation
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Naive Bayes Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# BERT Model for Text Classification
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y_train)), ignore_mismatched_sizes=True)

# Tokenizing text data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Convert labels to tensor
train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)

# Creating datasets
from datasets import Dataset
train_data = {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels}
test_data = {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': test_labels}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer for BERT model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Predictions and evaluation
predictions = trainer.predict(test_dataset)
y_pred = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
print("BERT Model Classification Report:")
print(classification_report(y_test, y_pred))

# Saving models
import joblib
joblib.dump(model, "news_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Reporting & Communication
"""
Key Insights:
1. **Naive Bayes vs BERT** - Naive Bayes is efficient but lacks deep contextual understanding; BERT achieves higher accuracy but requires more computational resources.
2. **Data Imbalance** - Some categories may have more data than others, influencing model performance.
3. **TF-IDF vs BERT Tokenization** - TF-IDF represents term frequency well, but BERT captures deeper semantic meaning.

Future Improvements:
- Implement hyperparameter tuning for both models to optimize performance.
- Experiment with other deep learning architectures like LSTMs or CNNs for text classification.
- Incorporate real-world datasets for more diverse language patterns.
- Deploy the best-performing model using a web-based API for real-time classification.
"""
