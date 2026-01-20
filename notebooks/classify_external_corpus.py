#!/usr/bin/env python3
"""
Classify External Corpus: CNN/Daily Mail 100 Texts
Classify texts with all three models and analyze agreement/confusion.
"""

import pandas as pd
import numpy as np
import joblib
import torch
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import spacy
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXTERNAL_DATA_PATH = "cnn_dailymail_100_texts.csv"
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
print(f"External Corpus Classification: CNN/Daily Mail 100 Texts")
print(f"{'='*80}\n")
print(f"Using device: {device}\n")


def load_data():
    """Load external corpus."""
    print("[1/6] Loading external corpus...")

    df = pd.read_csv(EXTERNAL_DATA_PATH, header=None, names=['text'])
    texts = df['text'].tolist()

    print(f"  Loaded {len(texts)} texts")
    print(f"  Avg length: {np.mean([len(t.split()) for t in texts]):.0f} words")
    return texts


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


def classify_tfidf(texts):
    """Classify with TF-IDF + LR."""
    print("\n[3/5] Classifying with TF-IDF + LR...")

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/tfidf_lr.pkl")

    texts_tfidf = vectorizer.transform(texts)
    predictions = model.predict(texts_tfidf)

    # Handle both string and numeric predictions
    le = joblib.load("models/bert_label_encoder.pkl")
    if isinstance(predictions[0], str):
        labels = predictions  # Already string labels
    else:
        labels = [le.classes_[int(p)] for p in predictions]

    print(f"  ✓ TF-IDF predictions completed")
    return labels


def classify_linguistic(texts):
    """Classify with Linguistic + RF."""
    print("\n[4/5] Classifying with Linguistic + RF...")

    model = joblib.load("models/linguistic_rf.pkl")
    nlp = spacy.load('en_core_web_sm')

    features_list = []
    for text in tqdm(texts, desc="  Extracting features"):
        features_list.append(extract_linguistic_features(text, nlp))

    features_df = pd.DataFrame(features_list)
    predictions = model.predict(features_df)

    # Handle both string and numeric predictions
    le = joblib.load("models/bert_label_encoder.pkl")
    if isinstance(predictions[0], str):
        labels = predictions  # Already string labels
    else:
        labels = [le.classes_[int(p)] for p in predictions]

    print(f"  ✓ Linguistic predictions completed")
    return labels


def classify_bert(texts):
    """Classify with BERT."""
    print("\n[5/5] Classifying with BERT...")

    # Load model
    model = BertForSequenceClassification.from_pretrained("models/bert_category_classifier").to(device)
    tokenizer = BertTokenizer.from_pretrained("models/bert_category_classifier")

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

    # Use BERT label encoder
    le = joblib.load("models/bert_label_encoder.pkl")
    labels = [le.classes_[int(p)] for p in predictions]

    print(f"  ✓ BERT predictions completed")
    return labels


def analyze_agreement(tfidf_preds, linguistic_preds, bert_preds, categories):
    """Analyze agreement between models."""
    print("\n[5/5] Analyzing agreement and creating visualizations...")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'text_id': range(100),
        'tfidf_pred': tfidf_preds,
        'linguistic_pred': linguistic_preds,
        'bert_pred': bert_preds
    })

    # Save results
    results_df.to_csv("results/external_corpus_predictions.csv", index=False)
    print("  ✓ Saved: results/external_corpus_predictions.csv")

    # Summary statistics
    print("\n" + "="*80)
    print("PREDICTION DISTRIBUTIONS")
    print("="*80)

    print("\nTF-IDF Predictions:")
    tfidf_counts = Counter(tfidf_preds)
    for cat in categories:
        print(f"  {cat:15s}: {tfidf_counts.get(cat, 0):3d} ({tfidf_counts.get(cat, 0)}%)")

    print("\nLinguistic Predictions:")
    linguistic_counts = Counter(linguistic_preds)
    for cat in categories:
        print(f"  {cat:15s}: {linguistic_counts.get(cat, 0):3d} ({linguistic_counts.get(cat, 0)}%)")

    print("\nBERT Predictions:")
    bert_counts = Counter(bert_preds)
    for cat in categories:
        print(f"  {cat:15s}: {bert_counts.get(cat, 0):3d} ({bert_counts.get(cat, 0)}%)")

    # Agreement analysis
    print("\n" + "="*80)
    print("MODEL AGREEMENT")
    print("="*80)

    tfidf_bert_agree = sum(1 for t, b in zip(tfidf_preds, bert_preds) if t == b)
    tfidf_linguistic_agree = sum(1 for t, l in zip(tfidf_preds, linguistic_preds) if t == l)
    bert_linguistic_agree = sum(1 for b, l in zip(bert_preds, linguistic_preds) if b == l)

    all_three_agree = sum(1 for t, l, b in zip(tfidf_preds, linguistic_preds, bert_preds) if t == l == b)

    print(f"\nTF-IDF ↔ BERT agreement: {tfidf_bert_agree}/100 ({tfidf_bert_agree}%)")
    print(f"TF-IDF ↔ Linguistic agreement: {tfidf_linguistic_agree}/100 ({tfidf_linguistic_agree}%)")
    print(f"BERT ↔ Linguistic agreement: {bert_linguistic_agree}/100 ({bert_linguistic_agree}%)")
    print(f"All three agree: {all_three_agree}/100 ({all_three_agree}%)")

    # Create confusion matrices between models
    print("\n" + "="*80)
    print("CONFUSION MATRICES (Model Comparisons)")
    print("="*80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TF-IDF vs BERT
    cm1 = confusion_matrix(
        [categories.index(p) for p in tfidf_preds],
        [categories.index(p) for p in bert_preds],
        labels=range(len(categories))
    )

    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=axes[0])
    axes[0].set_title('TF-IDF vs BERT', fontsize=14, weight='bold')
    axes[0].set_xlabel('BERT Prediction', fontsize=12)
    axes[0].set_ylabel('TF-IDF Prediction', fontsize=12)

    # TF-IDF vs Linguistic
    cm2 = confusion_matrix(
        [categories.index(p) for p in tfidf_preds],
        [categories.index(p) for p in linguistic_preds],
        labels=range(len(categories))
    )

    sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', xticklabels=categories, yticklabels=categories, ax=axes[1])
    axes[1].set_title('TF-IDF vs Linguistic', fontsize=14, weight='bold')
    axes[1].set_xlabel('Linguistic Prediction', fontsize=12)
    axes[1].set_ylabel('TF-IDF Prediction', fontsize=12)

    # BERT vs Linguistic
    cm3 = confusion_matrix(
        [categories.index(p) for p in bert_preds],
        [categories.index(p) for p in linguistic_preds],
        labels=range(len(categories))
    )

    sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens', xticklabels=categories, yticklabels=categories, ax=axes[2])
    axes[2].set_title('BERT vs Linguistic', fontsize=14, weight='bold')
    axes[2].set_xlabel('Linguistic Prediction', fontsize=12)
    axes[2].set_ylabel('BERT Prediction', fontsize=12)

    plt.tight_layout()
    plt.savefig("results/external_corpus_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print("\n  ✓ Saved: results/external_corpus_confusion_matrices.png")

    # Normalized confusion matrices (row-wise)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TF-IDF vs BERT (normalized)
    cm1_norm = cm1.astype('float') / cm1.sum(axis=1, keepdims=True)
    sns.heatmap(cm1_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('TF-IDF vs BERT (Normalized)', fontsize=14, weight='bold')
    axes[0].set_xlabel('BERT Prediction', fontsize=12)
    axes[0].set_ylabel('TF-IDF Prediction', fontsize=12)

    # TF-IDF vs Linguistic (normalized)
    cm2_norm = cm2.astype('float') / cm2.sum(axis=1, keepdims=True)
    sns.heatmap(cm2_norm, annot=True, fmt='.2f', cmap='Reds', xticklabels=categories, yticklabels=categories, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('TF-IDF vs Linguistic (Normalized)', fontsize=14, weight='bold')
    axes[1].set_xlabel('Linguistic Prediction', fontsize=12)
    axes[1].set_ylabel('TF-IDF Prediction', fontsize=12)

    # BERT vs Linguistic (normalized)
    cm3_norm = cm3.astype('float') / cm3.sum(axis=1, keepdims=True)
    sns.heatmap(cm3_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=categories, yticklabels=categories, ax=axes[2], vmin=0, vmax=1)
    axes[2].set_title('BERT vs Linguistic (Normalized)', fontsize=14, weight='bold')
    axes[2].set_xlabel('Linguistic Prediction', fontsize=12)
    axes[2].set_ylabel('BERT Prediction', fontsize=12)

    plt.tight_layout()
    plt.savefig("results/external_corpus_confusion_matrices_normalized.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/external_corpus_confusion_matrices_normalized.png")

    # Prediction distribution bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.25

    tfidf_vals = [tfidf_counts.get(cat, 0) for cat in categories]
    linguistic_vals = [linguistic_counts.get(cat, 0) for cat in categories]
    bert_vals = [bert_counts.get(cat, 0) for cat in categories]

    bars1 = ax.bar(x - width, tfidf_vals, width, label='TF-IDF', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, linguistic_vals, width, label='Linguistic', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, bert_vals, width, label='BERT', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Genre', fontsize=12, weight='bold')
    ax.set_ylabel('Count', fontsize=12, weight='bold')
    ax.set_title('Genre Prediction Distribution on External Corpus', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/external_corpus_distribution.png", dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/external_corpus_distribution.png")

    # Find disagreements
    print("\n" + "="*80)
    print("DISAGREEMENT ANALYSIS (Where models differ)")
    print("="*80)

    disagreements = []

    for i in range(len(texts)):
        tfidf_p = tfidf_preds[i]
        linguistic_p = linguistic_preds[i]
        bert_p = bert_preds[i]

        if not (tfidf_p == linguistic_p == bert_p):
            disagreements.append({
                'text_id': i,
                'tfidf': tfidf_p,
                'linguistic': linguistic_p,
                'bert': bert_p,
                'agreement': 'full' if tfidf_p == bert_p else 'partial'
            })

    print(f"\nTotal disagreements: {len(disagreements)}/100 ({len(disagreements)}%)")

    if disagreements:
        print("\nSample disagreements (first 10):")
        for d in disagreements[:10]:
            print(f"  Text {d['text_id']:3d}: TF-IDF={d['tfidf']:12s} Linguistic={d['linguistic']:12s} BERT={d['bert']:12s} [{d['agreement']}]")

    # Save disagreements
    disagreements_df = pd.DataFrame(disagreements)
    disagreements_df.to_csv("results/external_corpus_disagreements.csv", index=False)
    print("\n  ✓ Saved: results/external_corpus_disagreements.csv")

    print("\n" + "="*80)
    print("EXTERNAL CORPUS CLASSIFICATION COMPLETE")
    print("="*80)


def main():
    # Load data
    texts = load_data()

    # Load categories
    le = joblib.load("models/bert_label_encoder.pkl")
    categories = le.classes_.tolist()
    print(f"\n[2/5] Categories: {categories}")

    # Classify with all three models
    tfidf_labels = classify_tfidf(texts)
    linguistic_labels = classify_linguistic(texts)
    bert_labels = classify_bert(texts)

    # Analyze agreement
    analyze_agreement(tfidf_labels, linguistic_labels, bert_labels, categories)


if __name__ == "__main__":
    main()
