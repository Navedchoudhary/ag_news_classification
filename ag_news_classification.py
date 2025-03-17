# Import necessary libraries
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the AG News dataset from HuggingFace
dataset = load_dataset("ag_news")

# Display the first few entries of the dataset
# print(dataset['train'].head())

# Define a function to clean the text data
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the text column
dataset['train'] = dataset['train'].map(lambda x: {'text': clean_text(x['text']), 'label': x['label']})

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Save the processed data to a CSV file
df.to_csv('processed_ag_news.csv', index=False)

# Document cleaning decisions
# - Removed HTML tags to avoid noise in text.
# - Removed special characters and numbers to focus on words.
# - Converted text to lowercase for uniformity.

# Exploratory Data Analysis
# Load the processed data
df = pd.read_csv('processed_ag_news.csv')

# Display basic statistics
print(df.describe())

# Visualize the distribution of article categories
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Distribution of Article Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2, 3], labels=['World', 'Sports', 'Business', 'Science/Technology'])
plt.show()

# Create a word cloud for the most common words in the dataset
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of News Articles')
plt.show()

# Analyze the average length of articles by category
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='text_length', data=df)
plt.title('Article Length by Category')
plt.xlabel('Category')
plt.ylabel('Length of Article')
plt.xticks(ticks=[0, 1, 2, 3], labels=['World', 'Sports', 'Business', 'Science/Technology'])
plt.show()

# Language Model Classification
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# Tokenize the input data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Create a dataset class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = NewsDataset(train_encodings, y_train.tolist())
test_dataset = NewsDataset(test_encodings, y_test.tolist())

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

# Train the model
trainer.train()

# After training, you can use the model for predictions
