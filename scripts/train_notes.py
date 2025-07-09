# Loading packages that are going to be used
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load dataset
dataset = load_dataset("sms_spam")
# Data set used for training the model it only has one split, so we are splitting it later


# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Loads the DistilBERT tokenizer which converts text into token IDs that the BERT model understands


# Defines how to tokenize each SMS message in the Dataset
def tokenize_fn(example):
    return tokenizer(example["sms"], truncation=True, padding="max_length")


#Applies the tokenize function
encoded = dataset.map(tokenize_fn, batched=True)

# Manual train-test split
split_dataset = encoded["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
# This model is a smaller, faster version of BERT which is "Foundational NLP Model" from Google
# Has 97% of BERTs performance
# Good at Sentiment Analysis, Text Classification, Question Answering, Name Entity Recognition



# Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer
args = TrainingArguments("spam-detector", eval_strategy="epoch", per_device_train_batch_size=16)
# An Epoch is one time through the training dataset

# This is the actual step where the Neural Net is being run and evaluated
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# This is the actual step where the Neural Net is being run and evaluated



# Trains and saves model so it can be loaded in the streamlit app later.
trainer.train()
trainer.save_model("bot_classifier")
tokenizer.save_pretrained("bot_classifier")
