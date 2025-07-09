# Script for evaluating the trained model

# Importing necessary packages
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
import umap

# --- Load model and tokenizer ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint_dir = "../models/checkpoint-58174"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# --- Load test set ---
from data_utility import final_dataset

test_set = final_dataset["test"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Prediction Loop ---
true_labels, pred_labels, probs_list, texts = [], [], [], []
for example in test_set:
    text = example["text"]
    if not isinstance(text, str) or not text.strip() or "�" in text:
        continue
    try:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred = torch.argmax(probs).item()
        true_labels.append(example["label"])
        pred_labels.append(pred)
        probs_list.append(probs.cpu().numpy()[0])
        texts.append(text)
    except Exception as e:
        print(f"Skipping due to error: {e}")
        continue

# --- Save Predictions ---
output_dir = os.path.join("..", "outputs")
os.makedirs(output_dir, exist_ok=True)
probs_array = np.array(probs_list)
df = pd.DataFrame({
    "text": texts,
    "true_label": true_labels,
    "pred_label": pred_labels,
    "prob_0": probs_array[:, 0],
    "prob_1": probs_array[:, 1]
})
df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

# --- Metrics ---
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels)
rec = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

# metrics df
metrics_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1
}])
metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
print(metrics_df)
print("\nDetailed classification report:")
print(classification_report(true_labels, pred_labels, target_names=["Human", "Not Human"]))

# --- Plots (modular functions) ---
# Confusion Matrix
def plot_confusion_matrix():
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "Not Human"], yticklabels=["Human", "Not Human"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.show()

# ROC Curve
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(true_labels, probs_array[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    sns.lineplot(x=fpr, y=tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.show()

# Percision Recall Curve
def plot_precision_recall():
    precision, recall, _ = precision_recall_curve(true_labels, probs_array[:, 1])
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker=".")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.show()

# Confidence Distribution Plot
def plot_confidence_distribution():
    plt.figure(figsize=(6, 4))
    sns.histplot(probs_array[:, 1], bins=20, kde=True)
    plt.title("Histogram of Confidence Scores (Class 1)")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"))
    plt.show()

# Calibration PLot
def plot_calibration():
    prob_true, prob_pred = calibration_curve(true_labels, probs_array[:, 1], n_bins=10)
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Calibration Plot")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_plot.png"))
    plt.show()

# UMAP
def plot_umap():
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(probs_array)
    emb_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    emb_df["label"] = true_labels
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=emb_df, x="UMAP1", y="UMAP2", hue="label", palette="Set2", alpha=0.7)
    plt.title("UMAP Projection of Prediction Probabilities")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_embedding.png"))
    plt.show()

# --- Misclassifications ---
df["misclassified"] = df["true_label"] != df["pred_label"]
df[df["misclassified"]].to_csv(os.path.join(output_dir, "misclassified_examples.csv"), index=False)

# --- Run desired plots ---
plot_confusion_matrix()
plot_roc_curve()
plot_precision_recall()
plot_confidence_distribution()
plot_calibration()
plot_umap()
