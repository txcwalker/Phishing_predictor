# Code for referencing the final dataset to be able runthe evaluate model and Raw EDA scripts without training the entire
# dataset

import os

# Redirect cache to a dummy folder (e.g., inside project or external drive)
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/txcwa/Desktop/empty_cache"
os.environ["HF_HOME"] = "C:/Users/txcwa/Desktop/empty_cache"

# packages used for training
import pandas as pd
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch


# --- Load Datasets ---
sms = load_dataset("sms_spam")["train"]
persona = load_dataset("bavard/personachat_truecased", trust_remote_code=True)["train"] # 131,8
phish1 = load_dataset("zefang-liu/phishing-email-dataset")["train"]
phish2 = load_dataset("yashpapa6969/phising_attacks")["train"]
daily_dialog_train = load_dataset("pixelsandpointers/better_daily_dialog")["train"]
daily_dialog_test = load_dataset("pixelsandpointers/better_daily_dialog")["test"]
daily_dialog_validation = load_dataset("pixelsandpointers/better_daily_dialog")["validation"]
cnn_dm_train = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
cnn_dm_test = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]
cnn_dm_validation = load_dataset("abisee/cnn_dailymail", "3.0.0")["validation"]
spam_ds_train = load_dataset("distrib134/ultimate_spam_detection_2")['train']
spam_ds_test = load_dataset("distrib134/ultimate_spam_detection_2")['test']


# --- Labeling Functions ---

# SMS label function
def label_sms(example):
    return {"text": example["sms"], "label": int(0 if str(example["label"]) == "ham" else 1)}

# Persona label function
def extract_true_utterance(example):
    idx = example["utterance_idx"]
    if idx is not None and idx < len(example["candidates"]):
        return {"text": example["candidates"][idx], "label": 0}
    else:
        return {"text": "", "label": 0}

# Phishing 1 label function
def label_phish1(example):
    return {"text": example["Email Text"], "label": 1 if example["Email Type"] == "Phishing Email" else 0}

# Phishing 2 Label function
def label_phish2(example):
    return {"text": example["Body"], "label": 1 if example["Label"] == 1 else 0}

# Daily dialogue label function
def label_daily_dialog(example):
    return {"text": example["utterance"], "label": 0}

# CNN summarizer label function
def label_cnn(example):
    return {"text": example["highlights"], "label": 0}


# --- Apply Labels ---

sms_labeled = sms.map(label_sms).remove_columns(["sms"])
# Need to reorder columns for later
def reorder_columns(dataset):
    return dataset.map(lambda x: {"text": x["text"], "label": x["label"]}, remove_columns=dataset.column_names)

# Example for a single dataset
sms_reordered = reorder_columns(sms_labeled)
from datasets import ClassLabel, Value, Features

# Force label column to be int64
sms_labeled = sms_labeled.cast_column("label", Value("int64"))

# Applying labeling function to persona dataset
persona_labeled = persona.map(extract_true_utterance).remove_columns(
    ["personality", "candidates", "history", "conv_id", "utterance_idx"]
)
# Applying phish 1 and 2 labels
phish1_labeled = phish1.map(label_phish1).remove_columns([col for col in phish1.column_names if col != "text"])
phish2_labeled = phish2.map(label_phish2).remove_columns([col for col in phish2.column_names if col != "text"])


# Setting up Test, Train and Validation seperation for datasets
# Applying labeling function to dialogue dataset and making train, test and validation set
dialog_labeled = DatasetDict({
    "train": daily_dialog_train.map(label_daily_dialog).remove_columns(
        [col for col in daily_dialog_train.column_names if col not in ["text"]]
    ),
    "test": daily_dialog_test.map(label_daily_dialog).remove_columns(
        [col for col in daily_dialog_test.column_names if col not in ["text"]]
    ),
    "validation": daily_dialog_validation.map(label_daily_dialog).remove_columns(
        [col for col in daily_dialog_validation.column_names if col not in ["text"]]
    ),
})

# Applying labeling function to CNN dataset and making train, test and validation set
cnn_labeled = DatasetDict({
    "train": cnn_dm_train.map(label_cnn).remove_columns(
        [col for col in cnn_dm_train.column_names if col not in ["text"]]
    ),
    "test": cnn_dm_test.map(label_cnn).remove_columns(
        [col for col in cnn_dm_test.column_names if col not in ["text"]]
    ),
    "validation": cnn_dm_validation.map(label_cnn).remove_columns(
        [col for col in cnn_dm_validation.column_names if col not in ["text"]]
    ),
})

# Split SMS (only has train) → 70/20/10
sms_split = sms_labeled.train_test_split(test_size=0.30, seed=42)
sms_val_test = sms_split["test"].train_test_split(test_size=1/3, seed=42)
sms_dataset = DatasetDict({
    "train": sms_split["train"],
    "validation": sms_val_test["train"],
    "test": sms_val_test["test"]
})

# Persona (has train and validation) → use as-is, just rename keys
persona_dataset = DatasetDict({
    "train": persona_labeled,
    "validation": load_dataset("bavard/personachat_truecased", split="validation", trust_remote_code=True).map(extract_true_utterance)
})

# Persona originally has only train and validation, so we'll split train into 80/20
persona_train_full = persona_labeled.train_test_split(test_size=0.2, seed=42)
persona_validation_raw = load_dataset("bavard/personachat_truecased", split="validation", trust_remote_code=True)
persona_validation_labeled = persona_validation_raw.map(extract_true_utterance)
persona_dataset = DatasetDict({
    "train": persona_train_full["train"],
    "test": persona_train_full["test"],
    "validation": persona_validation_labeled
})


# Phish1 → 70/20/10
phish1_split = phish1_labeled.train_test_split(test_size=0.30, seed=42)
phish1_val_test = phish1_split["test"].train_test_split(test_size=1/3, seed=42)
phish1_dataset = DatasetDict({
    "train": phish1_split["train"],
    "validation": phish1_val_test["train"],
    "test": phish1_val_test["test"]
})

# Phish2 → 70/20/10
phish2_split = phish2_labeled.train_test_split(test_size=0.30, seed=42)
phish2_val_test = phish2_split["test"].train_test_split(test_size=1/3, seed=42)
phish2_dataset = DatasetDict({
    "train": phish2_split["train"],
    "validation": phish2_val_test["train"],
    "test": phish2_val_test["test"]
})

dialog_dataset = DatasetDict({
    "train": dialog_labeled["train"],
    "validation": dialog_labeled["validation"],
    "test": dialog_labeled["test"]
})

# CNN/DailyMail (has all three splits) → use as-is
cnn_dataset = DatasetDict({
    "train": cnn_labeled["train"],
    "validation": cnn_labeled["validation"],
    "test": cnn_labeled["test"]
})


#
spam_ds_split = spam_ds_train.train_test_split(test_size=0.1, seed=42)
spam_dataset = DatasetDict({
    "train": spam_ds_split['train'].rename_column("labels", "label"),
    "test": spam_ds_test.rename_column("labels", "label"),
    "validation": spam_ds_split['test'].rename_column("labels", "label")
})


from datasets import Value
# Force label column to be raw integers
for dataset in [sms_labeled, persona_labeled, phish1_labeled, phish2_labeled,
                dialog_labeled["train"], dialog_labeled["test"], dialog_labeled["validation"],
                cnn_labeled["train"], cnn_labeled["test"], cnn_labeled["validation"],
                spam_dataset["train"], spam_dataset["test"], spam_dataset["validation"]]:
    dataset = dataset.cast_column("label", Value("int64"))


# Combine all datasets by split
train_dataset = concatenate_datasets([
    sms_dataset["train"],
    persona_dataset["train"],
    phish1_dataset["train"],
    phish2_dataset["train"],
    dialog_dataset["train"],
    cnn_dataset["train"],
    spam_dataset["train"]

])

val_dataset = concatenate_datasets([
    sms_dataset["validation"],
    persona_dataset["validation"],
    phish1_dataset["validation"],
    phish2_dataset["validation"],
    dialog_dataset["validation"],
    cnn_dataset["validation"],
    spam_dataset["validation"]
])

test_dataset = concatenate_datasets([
    sms_dataset["test"],
    phish1_dataset["test"],
    phish2_dataset["test"],
    dialog_dataset["test"],
    cnn_dataset["test"],
    spam_dataset["test"]
])

# Final DatasetDict
final_dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Filter out null labels before tokenization
for split in final_dataset:
    final_dataset[split] = final_dataset[split].filter(lambda example: example["label"] is not None)

# --- Optional: Print Class Balance ---
for split in final_dataset:
    df = pd.DataFrame(final_dataset[split])
    print(f"{split.capitalize()} Split Class Balance:\n{df['label'].value_counts()}\n")

# --- Tokenizer ---
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(example):
    # Ensure all text inputs are strings
    texts = [str(t) if t is not None else "" for t in example["text"]]
    return tokenizer(texts, truncation=True, padding="max_length")


tokenized_datasets = final_dataset.map(tokenize_fn, batched=True)
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
    output_all_columns=False
)

# Explicitly convert labels to int64
for split in ["train", "validation", "test"]:
    tokenized_datasets[split] = tokenized_datasets[split].map(
        lambda x: {"label": int(x["label"])},
        batched=False
    )

# --- Model ---
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)