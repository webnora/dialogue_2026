#!/usr/bin/env python3
"""
Baseline Models Comparison
Compare TF-IDF+LR, Linguistic+RF, and BERT models on the same test set.

Generates:
- Comparison table with metrics
- Visualizations (bar charts, confusion matrices)
- McNemar's test for statistical significance
"""

import pandas as pd
import numpy as np
import json
import joblib
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import spacy
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Configuration
DATA_PATH = "data/cleaned_combined_guardian.csv"
MAX_LENGTH = 256
BATCH_SIZE = 16

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\n{'='*80}")
print(f"Baseline Models Comparison - Guardian Genre Classification")
print(f"{'='*80}\n")
print(f"Using device: {device}\n")


def load_data():
    """Load and prepare data."""
    print("[1/7] Loading data...")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["cleaned_text", "category"])

    # Clean data
    df["cleaned_text"] = (
        df["cleaned_text"]
        .astype(str)
        .replace({"nan": "", "None": "", "null": ""})
    )
    nan_mask = df["cleaned_text"].str.strip().isin(["", "nan", "None", "null"])
    df = df[~nan_mask]
    df["cleaned_text"] = df["cleaned_text"].astype(str)
    df["category"] = df["category"].astype(str)

    print(f"  Dataset size: {len(df)}")

    # Get texts and labels
    texts = df["cleaned_text"].tolist()
    categories = df["category"].tolist()

    # Load label encoder
    label_encoder = joblib.load("models/bert_label_encoder.pkl")
    labels = label_encoder.transform(categories)

    # Split data (same as training)
    from sklearn.model_selection import train_test_split

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    print(f"  Test set size: {len(test_texts)}")
    print(f"  Categories: {label_encoder.classes_.tolist()}")

    return test_texts, test_labels, label_encoder


def load_tfidf_model():
    """Load TF-IDF + LR model."""
    print("\n[2/7] Loading TF-IDF + LR model...")

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/tfidf_lr.pkl")

    print("  ✓ TF-IDF + LR loaded")
    return vectorizer, model


def load_linguistic_model():
    """Load Linguistic + RF model."""
    print("\n[3/7] Loading Linguistic + RF model...")

    model = joblib.load("models/linguistic_rf.pkl")
    nlp = spacy.load('en_core_web_sm')

    print("  ✓ Linguistic + RF loaded")
    return model, nlp


def extract_linguistic_features(text, nlp):
    """Extract linguistic features from text."""
    doc = nlp(text)
    features = {}

    tokens = [token.text.lower() for token in doc if token.is_alpha]
    features['type_token_ratio'] = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

    sentences = list(doc.sents)
    if len(sentences) > 0:
        sent_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = np.mean(sent_lengths)
    else:
        features['avg_sentence_length'] = 0

    first_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['i', 'we', 'me', 'us', 'my', 'our'])
    second_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['you', 'your'])
    third_person = sum(1 for token in doc if token.tag_ == 'PRP' and token.text.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their'])
    total_tokens = len(doc)

    if total_tokens > 0:
        features['first_person_ratio'] = first_person / total_tokens
        features['second_person_ratio'] = second_person / total_tokens
        features['third_person_ratio'] = third_person / total_tokens
    else:
        features['first_person_ratio'] = 0
        features['second_person_ratio'] = 0
        features['third_person_ratio'] = 0

    MODAL_VERBS = ['can', 'could', 'may', 'might', 'must', 'should', 'ought', 'would', 'shall']
    STANCE_MARKERS = ['arguably', 'reportedly', 'seemingly', 'apparently', 'undoubtedly', 'clearly', 'obviously', 'evidently']
    HEDGES = ['perhaps', 'possibly', 'somewhat', 'rather', 'quite', 'relatively', 'comparatively']
    REPORTING_VERBS = ['said', 'says', 'say', 'told', 'tells', 'tell', 'claimed', 'claims', 'claim', 'stated', 'states', 'state', 'reported', 'reports', 'report']

    modal_count = sum(1 for token in doc if token.text.lower() in MODAL_VERBS)
    features['modal_ratio'] = modal_count / total_tokens if total_tokens > 0 else 0

    stance_count = sum(1 for token in doc if token.text.lower() in STANCE_MARKERS)
    features['stance_markers_ratio'] = stance_count / total_tokens if total_tokens > 0 else 0

    hedge_count = sum(1 for token in doc if token.text.lower() in HEDGES)
    features['hedges_ratio'] = hedge_count / total_tokens if total_tokens > 0 else 0

    quote_count = text.count('"')
    features['quotes_ratio'] = quote_count / total_tokens if total_tokens > 0 else 0

    reporting_count = sum(1 for token in doc if token.text.lower() in REPORTING_VERBS)
    features['reporting_verbs_ratio'] = reporting_count / total_tokens if total_tokens > 0 else 0

    return features


def load_bert_model():
    """Load BERT model."""
    print("\n[4/7] Loading BERT model...")

    model = BertForSequenceClassification.from_pretrained("models/bert_category_classifier").to(device)
    tokenizer = BertTokenizer.from_pretrained("models/bert_category_classifier")

    print("  ✓ BERT loaded")
    return model, tokenizer


def predict_tfidf(texts, vectorizer, model):
    """Predict with TF-IDF + LR."""
    print("\n[5/7] Predicting with TF-IDF + LR...")

    texts_tfidf = vectorizer.transform(texts)
    predictions = model.predict(texts_tfidf)

    print(f"  ✓ TF-IDF predictions completed")
    return predictions


def predict_linguistic(texts, model, nlp):
    """Predict with Linguistic + RF."""
    print("\n[5/7] Predicting with Linguistic + RF...")

    features_list = [extract_linguistic_features(text, nlp) for text in tqdm(texts, desc="  Extracting features")]
    features_df = pd.DataFrame(features_list)
    predictions = model.predict(features_df)

    print(f"  ✓ Linguistic predictions completed")
    return predictions


def predict_bert(texts, model, tokenizer, label_encoder):
    """Predict with BERT."""
    print("\n[5/7] Predicting with BERT...")

    # Create dataset
    test_dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, max_length=MAX_LENGTH,
        pad_to_multiple_of=8, return_tensors="pt"
    )

    test_dataloader = DataLoader(tokenized_test, batch_size=BATCH_SIZE, collate_fn=data_collator)

    model.eval()
    predictions = []

    for batch in tqdm(test_dataloader, desc="  BERT inference"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1)
        predictions.extend(batch_preds.cpu().numpy())

    print(f"  ✓ BERT predictions completed")
    return np.array(predictions)


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate metrics for a model."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 (macro)": f1_score(y_true, y_pred, average='macro'),
        "Precision (macro)": precision_score(y_true, y_pred, average='macro'),
        "Recall (macro)": recall_score(y_true, y_pred, average='macro'),
    }
    return metrics


def perform_mcnemar_test(y_true, y_pred1, y_pred2, model1_name, model2_name):
    """Perform McNemar's test for statistical significance."""
    print(f"\n  McNemar's test: {model1_name} vs {model2_name}")

    # Find where predictions differ
    # b = Model 1 correct, Model 2 incorrect
    # c = Model 1 incorrect, Model 2 correct
    b = ((y_pred1 == y_true) & (y_pred2 != y_true)).sum()
    c = ((y_pred1 != y_true) & (y_pred2 == y_true)).sum()

    print(f"  Model 1 correct, Model 2 incorrect: {b}")
    print(f"  Model 1 incorrect, Model 2 correct: {c}")

    # Perform McNemar's test with continuity correction
    if b + c > 0:
        # Chi-squared with continuity correction
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        p_value = 1 - chi2.cdf(chi2_stat, 1)

        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"

        print(f"  Chi2 statistic: {chi2_stat:.4f}")
        print(f"  p-value: {p_value:.4e} {significance}")

        return {
            "Comparison": f"{model1_name} vs {model2_name}",
            "b": int(b),
            "c": int(c),
            "chi2": float(chi2_stat),
            "p-value": float(p_value),
            "Significant": bool(p_value < 0.05)
        }
    else:
        print(f"  No disagreements between models")
        return {
            "Comparison": f"{model1_name} vs {model2_name}",
            "b": 0,
            "c": 0,
            "chi2": 0.0,
            "p-value": 1.0,
            "Significant": False
        }


def plot_comparison_table(metrics_df, mcnemar_results):
    """Create comparison table and visualizations."""
    print("\n[6/7] Creating visualizations...")

    os.makedirs("results", exist_ok=True)

    # 1. Comparison Table
    print("  Creating comparison table...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Round metrics for display
    display_df = metrics_df.copy()
    for col in ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center', colColours=['#f3f3f3']*len(display_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Highlight header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4a90e2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title("Baseline Models Comparison", fontsize=16, weight='bold', pad=20)
    plt.savefig("results/comparison_table.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/comparison_table.png")

    # 2. Bar chart comparison
    print("  Creating bar chart...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, weight='bold')

    metrics_to_plot = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]

        models = metrics_df["Model"].tolist()
        values = metrics_df[metric].tolist()

        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

        ax.set_ylabel(metric, fontsize=12, weight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(models, rotation=15, ha='right')

        # Add horizontal line at 0.8
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("results/comparison_barchart.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/comparison_barchart.png")

    # 3. McNemar's test results
    print("  Creating McNemar's test visualization...")
    if mcnemar_results:
        mcnemar_df = pd.DataFrame(mcnemar_results)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        # Select columns to display
        display_mcnemar = mcnemar_df[["Comparison", "b", "c", "chi2", "p-value", "Significant"]].copy()
        display_mcnemar["p-value"] = display_mcnemar["p-value"].apply(lambda x: f"{x:.4e}")
        display_mcnemar["Significant"] = display_mcnemar["Significant"].apply(lambda x: "Yes" if x else "No")

        table = ax.table(cellText=display_mcnemar.values, colLabels=display_mcnemar.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

        # Highlight header
        for i in range(len(display_mcnemar.columns)):
            table[(0, i)].set_facecolor('#9b59b6')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title("McNemar's Test Results (Statistical Significance)", fontsize=14, weight='bold', pad=20)
        plt.savefig("results/mcnemar_test.png", dpi=300, bbox_inches='tight')
        print("  ✓ Saved: results/mcnemar_test.png")


def save_results(metrics_df, mcnemar_results, confusion_matrices):
    """Save results to files."""
    print("\n[7/7] Saving results...")

    # Save metrics
    metrics_df.to_csv("results/comparison_metrics.csv", index=False)
    print("  ✓ Saved: results/comparison_metrics.csv")

    # Save McNemar results
    if mcnemar_results:
        mcnemar_df = pd.DataFrame(mcnemar_results)
        mcnemar_df.to_csv("results/mcnemar_results.csv", index=False)
        print("  ✓ Saved: results/mcnemar_results.csv")

    # Save confusion matrices
    np.save("results/comparison_confusion_matrices.npy", confusion_matrices)
    print("  ✓ Saved: results/comparison_confusion_matrices.npy")

    # Create summary report
    report = {
        "n_models": len(metrics_df),
        "n_test_samples": 5000,
        "best_model": metrics_df.loc[metrics_df["Accuracy"].idxmax(), "Model"],
        "best_accuracy": float(metrics_df["Accuracy"].max()),
        "models": metrics_df.to_dict('records'),
        "mcnemar_tests": mcnemar_results if mcnemar_results else []
    }

    with open("results/comparison_summary.json", 'w') as f:
        json.dump(report, f, indent=2)

    print("  ✓ Saved: results/comparison_summary.json")


def main():
    # Load data
    test_texts, test_labels, label_encoder = load_data()

    # Load models
    tfidf_vectorizer, tfidf_model = load_tfidf_model()
    linguistic_model, nlp = load_linguistic_model()
    bert_model, bert_tokenizer = load_bert_model()

    # Make predictions
    tfidf_preds = predict_tfidf(test_texts, tfidf_vectorizer, tfidf_model)
    linguistic_preds = predict_linguistic(test_texts, linguistic_model, nlp)
    bert_preds = predict_bert(test_texts, bert_model, bert_tokenizer, label_encoder)

    # Calculate metrics
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)

    metrics_list = []
    metrics_list.append(calculate_metrics(test_labels, tfidf_preds, "TF-IDF + LR"))
    metrics_list.append(calculate_metrics(test_labels, linguistic_preds, "Linguistic + RF"))
    metrics_list.append(calculate_metrics(test_labels, bert_preds, "BERT"))

    metrics_df = pd.DataFrame(metrics_list)

    print("\n" + str(metrics_df.to_string(index=False)))

    # McNemar's tests
    print("\n" + "="*80)
    print("MCNEMAR'S TEST")
    print("="*80)

    mcnemar_results = []
    mcnemar_results.append(perform_mcnemar_test(test_labels, tfidf_preds, bert_preds, "TF-IDF", "BERT"))
    mcnemar_results.append(perform_mcnemar_test(test_labels, linguistic_preds, bert_preds, "Linguistic", "BERT"))
    mcnemar_results.append(perform_mcnemar_test(test_labels, tfidf_preds, linguistic_preds, "TF-IDF", "Linguistic"))

    # Confusion matrices
    confusion_matrices = {
        "tfidf": confusion_matrix(test_labels, tfidf_preds),
        "linguistic": confusion_matrix(test_labels, linguistic_preds),
        "bert": confusion_matrix(test_labels, bert_preds)
    }

    # Visualizations and save results
    plot_comparison_table(metrics_df, mcnemar_results)
    save_results(metrics_df, mcnemar_results, confusion_matrices)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nBest model: {metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']}")
    print(f"Best accuracy: {metrics_df['Accuracy'].max():.4f} ({metrics_df['Accuracy'].max()*100:.2f}%)")
    print("\nAll results saved to results/ directory")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
